import argparse
import functools
import bitsandbytes as bnb
import math 
from datasets import load_dataset,concatenate_datasets
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments,AutoModel, AutoTokenizer
from datasets import load_dataset, Dataset
from transformers import EarlyStoppingCallback
import json
import random
import wandb 
from huggingface_hub import HfApi
from sklearn.metrics.pairwise import cosine_similarity
#from sentence_transformers import SentenceTransformer
import logging
import numpy as np
import functools
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
# print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
# print("CUDA Version:", torch.version.cuda)
# print("PyTorch Version:", torch.__version__)

#top k=1 -> 1 relevant examples
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
wandb.init(project="random_relevant_zeroPrompt", entity="anant7")


# Load a pre-trained model for embeddings (e.g., sentence-transformers)
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = AutoModel.from_pretrained(embedding_model_name)
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

@functools.lru_cache(maxsize=None)
def get_embeddings(text):
    inputs = embedding_tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embedding = embedding_model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
    return embedding[0]

def precompute_relevant_examples(dataset, few_shot_examples, top_k=1
                                 , context_weight=0.5, question_weight=0.5):
    relevant_examples_map = {}
    
    # Pre-compute embeddings for few-shot examples
    few_shot_context_embeddings = np.array([get_embeddings(ex["context"]) for ex in few_shot_examples])
    few_shot_question_embeddings = np.array([get_embeddings(ex["question"]) for ex in few_shot_examples])
    
    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc="Pre-computing relevant examples"):
        context_embedding = get_embeddings(item['context'])
        question_embedding = get_embeddings(item['question'])
        
        context_similarities = cosine_similarity([context_embedding], few_shot_context_embeddings)[0]
        question_similarities = cosine_similarity([question_embedding], few_shot_question_embeddings)[0]
        
        combined_similarities = (context_weight * context_similarities) + (question_weight * question_similarities)
        most_relevant_indices = np.argsort(combined_similarities)[-top_k:][::-1]
        
        relevant_examples_map[str(idx)] = most_relevant_indices.tolist()
    
    return relevant_examples_map

# Save the pre-computed relevant examples
def save_relevant_examples(relevant_examples_map, filename):
    with open(filename, 'w') as f:
        json.dump(relevant_examples_map, f)

# Load the pre-computed relevant examples
def load_relevant_examples(filename):
    with open(filename, 'r') as f:
        return json.load(f)

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

def load_few_shot_examples(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

few_shot_examples = load_few_shot_examples('few_shot_examples.json')

few_shot_prompt = """You are a financial expert assistant. Answer the following question based on the given context. Use step-by-step reasoning and provide numerical calculations where applicable. If asked to extract subject and object in relation, return only the single most relevant and applicable extraction. Use '\n' for line breaks in your response.

Examples:
{examples}

Now, please answer the following:

Context: {context}
Question: {question}
Answer (including calculation steps and final answer, or single most relevant relation extraction, use '\n' for line breaks):"""

def format_few_shot_examples(examples):
    formatted = ""
    for i, example in enumerate(examples, 1):
        formatted += f"Example {i}:\n"
        formatted += f"Context: {example['context']}\n"
        formatted += f"Question: {example['question']}\n"
        formatted += f"Answer: {example['answer']}\n\n"
    return formatted

#EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def create_prompt_formats(examples, few_shot_examples):
    texts = []

    for instruction, input, output, relevant_indices in zip(examples["context"], examples["question"], examples["answer"], examples["relevant_examples"]):
        # Decide randomly which type of examples to include
        choice = random.choices(
            ["both", "relevant_only", "random_only", "none"], 
            weights=[0.25, 0.25, 0.25, 0.25],  
            k=1)[0]

        if choice == "both":
            relevant_examples = [few_shot_examples[i] for i in relevant_indices]
            random_examples = random.sample(few_shot_examples, min(len(few_shot_examples), len(relevant_indices)))
            combined_examples = relevant_examples + random_examples
        elif choice == "relevant_only":
            combined_examples = [few_shot_examples[i] for i in relevant_indices]
        elif choice == "random_only":
            combined_examples = random.sample(few_shot_examples, min(len(few_shot_examples), len(relevant_indices)))
        else:
            combined_examples = []

        formatted_examples = format_few_shot_examples(combined_examples)
        
        text = few_shot_prompt.format(
            examples=formatted_examples,
            context=instruction,
            question=input
        ) + output
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

def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed: int, dataset: Dataset, few_shot_examples, num_proc=4):
    logger.info("Preprocessing dataset...")
    
    dataset = dataset.map(
        lambda examples: create_prompt_formats(examples, few_shot_examples),
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names
    )

    if len(dataset) == 0:
        raise ValueError("The dataset is empty after formatting prompts")

    logger.info("Tokenizing dataset...")
    _preprocessing_function = functools.partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)

    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names
    )

    logger.info("Filtering dataset...")
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    
    if len(dataset) == 0:
        raise ValueError("The dataset is empty after filtering")

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

def prepare_dataset(tokenizer, max_length, seed, few_shot_examples, top_k=1, context_weight=0.5, question_weight=0.5):
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

    # Pre-compute relevant examples
    relevant_examples_file = "relevant_examples.json"
    if not os.path.exists(relevant_examples_file):
        relevant_examples_map = precompute_relevant_examples(train_dataset, few_shot_examples, top_k, context_weight, question_weight)
        save_relevant_examples(relevant_examples_map, relevant_examples_file)
    else:
        relevant_examples_map = load_relevant_examples(relevant_examples_file)

    # Add relevant examples to the dataset
    train_dataset = train_dataset.add_column("relevant_examples", [relevant_examples_map.get(str(i), []) for i in range(len(train_dataset))])
    eval_dataset = eval_dataset.add_column("relevant_examples", [relevant_examples_map.get(str(i), []) for i in range(len(eval_dataset))])
    
    train_dataset = preprocess_dataset(tokenizer, max_length, seed, train_dataset, few_shot_examples)
    eval_dataset = preprocess_dataset(tokenizer, max_length, seed, eval_dataset, few_shot_examples)
    
    return train_dataset, eval_dataset

def train(model, tokenizer, train_dataset, eval_dataset, output_dir, resume_from_checkpoint=None):
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    modules = find_all_linear_names(model)
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)
    
    print_trainable_parameters(model)
    
    ddp_model = DDP(model, device_ids=[int(os.environ.get("LOCAL_RANK") or 0)])
    
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
        resume_from_checkpoint=resume_from_checkpoint
    )
    
    #train_sampler = DistributedSampler(dataset)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    ddp_model.module.config.use_cache = False
    
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint) #resume_from_checkpoint specify in command with op directory
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
    parser = argparse.ArgumentParser(description="Train the model with checkpoint resuming capability")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to the checkpoint to resume from")
    args = parser.parse_args()

    model_name = "meta-llama/Meta-Llama-3-8B"  
    bnb_config = create_bnb_config()
    model, tokenizer = load_model(model_name, bnb_config)
    max_length = get_max_length(model)

    few_shot_examples = load_few_shot_examples('few_shot_examples.json')

    # Prepare dataset before initializing distributed training
    train_dataset, eval_dataset = prepare_dataset(tokenizer, max_length, seed=9, few_shot_examples=few_shot_examples, top_k=1, context_weight=0.5, question_weight=0.5)
    
    # Initialize distributed training
    dist.init_process_group(backend="nccl")
    
    output_dir = "random_relevant_zeroPrompt_ResumeCheckpoints"
    train(model, tokenizer, train_dataset, eval_dataset, output_dir, resume_from_checkpoint=args.resume_from_checkpoint)
    
    if dist.get_rank() == 0:
        if os.path.exists(output_dir):
            print("Model file exists, proceeding with loading and merging...")
            try:
                model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
                model = model.merge_and_unload()
                output_merged_dir = "random_relevant_zeroPrompt/final_merged_checkpoint"
                os.makedirs(output_merged_dir, exist_ok=True)
                model.save_pretrained(output_merged_dir, safe_serialization=True)
                tokenizer.save_pretrained(output_merged_dir)
                print(f"Model merged and saved in {output_merged_dir}")
            except Exception as e:
                print(f"Error during model loading or merging: {e}")
        else:
            print("Model file does not exist. Please check the training and saving steps.")
    dist.destroy_process_group()
    wandb.finish()
#with checkpoint - fewshot_relevant_examples_llama3_2407.py --resume_from_checkpoint /path/to/checkpoint
#top -u username --> clear wandb process
#nohup torchrun --nproc_per_node=8 random_relevant_zeroPrompt.py > rrzp.log 2>&1 &
