import argparse
import functools
import bitsandbytes as bnb
import math
from datasets import load_dataset, concatenate_datasets
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
import torch.distributed as dist
from transformers import EarlyStoppingCallback
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments, AutoModel, AutoTokenizer
from datasets import load_dataset, Dataset
import json
import random
import wandb
from huggingface_hub import HfApi
from sklearn.metrics.pairwise import cosine_similarity
import logging
import numpy as np
import functools
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment setup
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")

wandb_api_key = os.getenv("WANDB_API_KEY")
if not wandb_api_key:
    raise ValueError("wandb API key not found. Please set the WANDB_API_KEY environment variable.")

wandb.login(key=wandb_api_key)
wandb.init(project="chain_of_thought_llama3", entity="anant7")

def load_model(model_name, bnb_config):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

cot_prompt = """You are a financial expert assistant. Answer the following question based on the given context. It is CRUCIAL that you use step-by-step reasoning and provide detailed numerical calculations where applicable. Break down your answer into clear, numbered steps. If asked to extract subject and object in relation, return only the single most relevant and applicable extraction. Use '\n' for line breaks in your response.

Context: {context}
Question: {question}
Answer (including calculation steps and final answer, or single most relevant relation extraction, use '\n' for line breaks):"""

def create_prompt_formats(examples):
    texts = []
    for instruction, input, output in zip(examples["context"], examples["question"], examples["answer"]):
        full_text = cot_prompt.format(
            context=instruction,
            question=input
        ) + output
        texts.append(full_text)
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

def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed: int, dataset: Dataset, num_proc=4):
    logger.info("Preprocessing dataset...")
    
    dataset = dataset.map(
        create_prompt_formats,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names
    )

    logger.info("Tokenizing dataset...")
    dataset = dataset.map(
        lambda batch: preprocess_batch(batch, tokenizer, max_length),
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names
    )

    logger.info("Filtering dataset...")
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    logger.info("Shuffling dataset...")
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
    eval_data = combined_data[train_size:]

    train_dataset = Dataset.from_dict({
        "context": [item["context"] for item in train_data],
        "question": [item["question"] for item in train_data],
        "answer": [item["answer"] for item in train_data],
    })

    eval_dataset = Dataset.from_dict({
        "context": [item["context"] for item in eval_data],
        "question": [item["question"] for item in eval_data],
        "answer": [item["answer"] for item in eval_data],
    })
    
    train_dataset = preprocess_dataset(tokenizer, max_length, seed, train_dataset)
    eval_dataset = preprocess_dataset(tokenizer, max_length, seed, eval_dataset)
    
    return train_dataset, eval_dataset

def train(model, tokenizer, train_dataset, eval_dataset, output_dir, resume_from_checkpoint=None):
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    modules = find_all_linear_names(model)
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)
    
    print_trainable_parameters(model)
    
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=12,
        warmup_steps=2,
        #max_steps=100,
        num_train_epochs=5,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        output_dir=output_dir,
        optim="paged_adamw_8bit",
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        report_to="wandb",
        weight_decay=0.01,
        load_best_model_at_end=True,
        save_total_limit=3,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    model.config.use_cache = False
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    trainer.save_model(output_dir)
    
    return trainer

def load_and_merge_model(output_dir):
    model = AutoPeftModelForCausalLM.from_pretrained(
        output_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    merged_model = model.merge_and_unload()
    return merged_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model with checkpoint resuming capability")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to the checkpoint to resume from")
    args = parser.parse_args()

    model_name = "meta-llama/Meta-Llama-3-8B"
    bnb_config = create_bnb_config()
    model, tokenizer = load_model(model_name, bnb_config)
    max_length = get_max_length(model)

    train_dataset, eval_dataset = prepare_dataset(tokenizer, max_length, seed=9)
    
    output_dir = "cot_llama3"
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    torch.cuda.set_device(local_rank)
    
    trainer = train(model, tokenizer, train_dataset, eval_dataset, output_dir, resume_from_checkpoint=args.resume_from_checkpoint)
    
    if dist.get_rank() == 0:
        merged_model = load_and_merge_model(output_dir)
        output_merged_dir = os.path.join(output_dir, "final_merged_checkpoint")
        merged_model.save_pretrained(output_merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_merged_dir)
        print(f"Merged model saved in {output_merged_dir}")
    
    dist.destroy_process_group()

#torchrun --nproc_per_node=8 cot_ft_l3.py --resume_from_checkpoint /home/araj/cot_llama3/checkpoint-655
#nohup torchrun --nproc_per_node=8 cot_ft_l3.py > cot_ft_l3.log 2>&1 &
#nohup python cot_inference.py > cot_inference.log 2>&1 &
