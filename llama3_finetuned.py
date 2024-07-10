import argparse
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import json
import random
import wandb 
from huggingface_hub import HfApi

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")

api = HfApi()
user = api.whoami(token=hf_token)
print(f"Logged in as: {user['name']}")

# Set up wandb
wandb_api_key = os.getenv("WANDB_API_KEY")
if not wandb_api_key:
    raise ValueError("wandb API key not found. Please set the WANDB_API_KEY environment variable.")

wandb.login(key=wandb_api_key)

# Initialize wandb
wandb.init(project="llama3_finetuned_0807", entity="anant7")


def load_model(model_name, bnb_config):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    tokenizer.pad_token = tokenizer.eos_token
    global EOS_TOKEN
    EOS_TOKEN = tokenizer.eos_token

    return model, tokenizer


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

#EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def create_prompt_formats(examples):
    instructions = examples["context"]
    inputs       = examples["question"]
    outputs      = examples["answer"]
    texts = []    
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(
            context=instruction,
            question=input
        ) + output + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max length: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length

def preprocess_batch(batch, tokenizer, max_length):
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )

def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed: int, dataset: Dataset):
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats, batched=True,)
    
    if len(dataset) == 0:
        raise ValueError("The dataset is empty after formatting prompts")

    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["context", "question", "answer"],
    )

    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    
    if len(dataset) == 0:
        raise ValueError("The dataset is empty after filtering")

    dataset = dataset.shuffle(seed=seed)

    return dataset

def create_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def create_peft_config(modules):
    return LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def print_trainable_parameters(model, use_4bit=False):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )

def prepare_dataset(tokenizer, max_length, seed):
    file_path = "merged_data.json"
    with open(file_path, "r") as f:
        combined_data = json.load(f)
    
    sample_size = len(combined_data) // 3
    combined_data = random.sample(combined_data, sample_size)

    train_size = int(0.9 * len(combined_data))
    train_data = combined_data[:train_size]

    data_dict = {
        "context": [item["context"] for item in train_data],
        "question": [item["question"] for item in train_data],
        "answer": [item["answer"] for item in train_data],
    }
    dataset = Dataset.from_dict(data_dict)
    
    dataset = preprocess_dataset(tokenizer, max_length, seed, dataset)
    return dataset

def train(model, tokenizer, dataset, output_dir):
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    modules = find_all_linear_names(model)
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)
    
    print_trainable_parameters(model)
    
    ddp_model = DDP(model, device_ids=[int(os.environ.get("LOCAL_RANK") or 0)])
    
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        #max_steps=100,
        num_train_epochs=5,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
    )
    
    #train_sampler = DistributedSampler(dataset)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    ddp_model.module.config.use_cache = False
    
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print(metrics)    
    
    if dist.get_rank() == 0:
        print("Saving last checkpoint of the model...")
        os.makedirs(output_dir, exist_ok=True)
        save_model = ddp_model.module if hasattr(ddp_model, 'module') else ddp_model
        try:
            save_model.save_pretrained(output_dir)
            print(f"Model saved in {output_dir}")
        except Exception as e:
            print(f"Failed to save model: {e}")
    
    dist.barrier()
    
    del ddp_model
    del trainer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3-8B" 
    bnb_config = create_bnb_config()
    model, tokenizer = load_model(model_name, bnb_config)
    max_length = get_max_length(model)
    
    # Prepare dataset before initializing distributed training
    dataset = prepare_dataset(tokenizer, max_length,9)
    
    # Initialize distributed training
    dist.init_process_group(backend="nccl")
    
    output_dir = "llama3_0807"
    train(model, tokenizer, dataset, output_dir)
    
    if dist.get_rank() == 0:
        if os.path.exists(output_dir):
            print("Model file exists, proceeding with loading and merging...")
            try:
                model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
                model = model.merge_and_unload()
                output_merged_dir = "llama3_0807/final_merged_checkpoint"
                os.makedirs(output_merged_dir, exist_ok=True)
                model.save_pretrained(output_merged_dir, safe_serialization=True)
                tokenizer.save_pretrained(output_merged_dir)
                print(f"Model merged and saved in {output_merged_dir}")
            except Exception as e:
                print(f"Error during model loading or merging: {e}")
        else:
            print("Model file does not exist. Please check the training and saving steps.")
    dist.destroy_process_group()