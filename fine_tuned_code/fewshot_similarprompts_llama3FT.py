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
wandb.init(project="fewshot_similarexamples_llama3", entity="anant7")


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

def precompute_relevant_examples(dataset, few_shot_examples, top_k=1, context_weight=0.5, question_weight=0.5):
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


#add few shot prompting

few_shot_examples = [
    {
        "context": "The Company\u2019s estimated future benefit payments as of December 31, 2019 are as follows: The Company has certain defined contribution plans, which accrue benefits for employees on a pro-rata basis during their employment period based on their individual salaries. The Company\u2019s accrued benefits related to defined contribution pension plans of $20 million as of December 31, 2019 and $18 million as of December 31, 2018. The annual cost of these plans amounted to approximately $86 million in 2019, $84 million in 2018 and $77 million in 2017.\nTable:\nYears\tPension Benefits\tOther Long Term Benefits\n2020\t32\t7\n2021\t29\t7\n2022\t32\t5\n2023\t41\t6\n2024\t51\t9\nFrom 2025 to 2029\t272\t35",
        "question": "What was the annual cost of contribution pension plans in 2019?",
        "answer": "Calculation:\nThe annual cost of contribution pension plans in 2019 is directly stated in the context.\n\nAnswer: $86 million"
    },
    {
        "context": "there is no goodwill assigned to reporting units within the balance sheet management segment . the following table shows the amount of goodwill allocated to each of the reporting units and the fair value as a percentage of book value for the reporting units in the trading and investing segment ( dollars in millions ) : . <table class='wikitable'><tr><td>1</td><td>reporting unit</td><td>december 31 2012 goodwill</td><td>december 31 2012 % (  % ) of fair value to book value</td></tr><tr><td>2</td><td>retail brokerage</td><td>$ 1791.8</td><td>190% ( 190 % )</td></tr><tr><td>3</td><td>market making</td><td>142.4</td><td>115% ( 115 % )</td></tr><tr><td>4</td><td>total goodwill</td><td>$ 1934.2</td><td>-</td></tr></table> we also evaluate the remaining useful lives on intangible assets each reporting period to determine whether events and circumstances warrant a revision to the remaining period of amortization . other intangible assets have a weighted average remaining useful life of 13 years . we did not recognize impairment on our other intangible assets in the periods presented . effects if actual results differ if our estimates of fair value for the reporting units change due to changes in our business or other factors , we may determine that an impairment charge is necessary . estimates of fair value are determined based on a complex model using estimated future cash flows and company comparisons . if actual cash flows are less than estimated future cash flows used in the annual assessment , then goodwill would have to be tested for impairment . the estimated fair value of the market making reporting unit as a percentage of book value was approximately 115% ( 115 % ) ; therefore , if actual cash flows are less than our estimated cash flows , goodwill impairment could occur in the market making reporting unit in the future . these cash flows will be monitored closely to determine if a further evaluation of potential impairment is necessary so that impairment could be recognized in a timely manner . in addition , following the review of order handling practices and pricing for order flow between e*trade securities llc and gi execution services , llc , our regulators may initiate investigations into our historical practices which could subject us to monetary penalties and cease-and-desist orders , which could also prompt claims by customers of e*trade securities llc . any of these actions could materially and adversely affect our market making and trade execution businesses , which could impact future cash flows and could result in goodwill impairment . intangible assets are amortized over their estimated useful lives . if changes in the estimated underlying revenue occur , impairment or a change in the remaining life may need to be recognized . estimates of effective tax rates , deferred taxes and valuation allowance description in preparing the consolidated financial statements , we calculate income tax expense ( benefit ) based on our interpretation of the tax laws in the various jurisdictions where we conduct business . this requires us to estimate current tax obligations and the realizability of uncertain tax positions and to assess temporary differences between the financial statement carrying amounts and the tax basis of assets and liabilities . these differences result in deferred tax assets and liabilities , the net amount of which we show as other assets or other liabilities on the consolidated balance sheet . we must also assess the likelihood that each of the deferred tax assets will be realized . to the extent we believe that realization is not more likely than not , we establish a valuation allowance . when we establish a valuation allowance or increase this allowance in a reporting period , we generally record a corresponding tax expense in the consolidated statement of income ( loss ) . conversely , to the extent circumstances indicate that a valuation allowance is no longer necessary , that portion of the valuation allowance is reversed , which generally reduces overall income tax expense . at december 31 , 2012 we had net deferred tax assets of $ 1416.2 million , net of a valuation allowance ( on state , foreign country and charitable contribution deferred tax assets ) of $ 97.8 million. .\nQuestion: what is the goodwill related to retail brokerage in 2012?\nAnswer: 1791.8\nQuestion: what about the total goodwill?\nAnswer: 1934.2\nQuestion: what portion of goodwill is related to retail brokerage?\n",
        "question": "Read the following texts and table with financial data from an S&P 500 earnings report carefully.Based on the question-answer history (if provided), answer the last question. The answer may require mathematical calculation based on the data provided.\nAnswer the last question in the given context, if history is provided for previous questions-answers use that information (if required based on the context) to get the answer for the last question",
        "answer": "Calculation:\n1. Retail brokerage goodwill: $1791.8 million\n2. Total goodwill: $1934.2 million\n3. Portion = Retail brokerage goodwill / Total goodwill\n4. 1791.8 / 1934.2 = 0.92638\n\nAnswer: 0.92638"
    },
    {
        "context": "We have got a Cognos if you remember back in the days that Cognos we had the original, very profitable application development tools business that really funded the growth of the business intelligence group, which became over time a larger part of the business.",
        "question": "Given phrases that describe the relationship between two words/phrases as options, extract the word/phrase pair and the corresponding lexical relationship between them from the input text. The output format should be \"relation1: word1, word2; relation2: word3, word4\". Options: product/material produced, manufacturer, distributed by, industry, position held, original broadcaster, owned by, founded by, distribution format, headquarters location, stock exchange, currency, parent organization, chief executive officer, director/manager, owner of, operator, member of, employer, chairperson, platform, subsidiary, legal form, publisher, developer, brand, business division, location of formation, creator.",
        "answer": "industry: Cognos, business intelligence"
    },
    {
        "context": "entergy texas , inc . and subsidiaries management 2019s financial discussion and analysis results of operations net income 2017 compared to 2016 net income decreased $ 31.4 million primarily due to lower net revenue , higher depreciation and amortization expenses , higher other operation and maintenance expenses , and higher taxes other than income taxes . 2016 compared to 2015 net income increased $ 37.9 million primarily due to lower other operation and maintenance expenses , the asset write-off of its receivable associated with the spindletop gas storage facility in 2015 , and higher net revenue . net revenue 2017 compared to 2016 net revenue consists of operating revenues net of : 1 ) fuel , fuel-related expenses , and gas purchased for resale , 2 ) purchased power expenses , and 3 ) other regulatory charges . following is an analysis of the change in net revenue comparing 2017 to 2016 . amount ( in millions ) .\n\nthe net wholesale revenue variance is primarily due to lower net capacity revenues resulting from the termination of the purchased power agreements between entergy louisiana and entergy texas in august 2016 . the purchased power capacity variance is primarily due to increased expenses due to capacity cost changes for ongoing purchased power capacity contracts . the transmission revenue variance is primarily due to a decrease in the amount of transmission revenues allocated by miso . the reserve equalization variance is due to the absence of reserve equalization expenses in 2017 as a result of entergy texas 2019s exit from the system agreement in august 2016 . see note 2 to the financial statements for a discussion of the system agreement. .\n\nTable:\n\tamount ( in millions )\n2016 net revenue\t$ 644.2\nnet wholesale revenue\t-35.1 ( 35.1 )\npurchased power capacity\t-5.9 ( 5.9 )\ntransmission revenue\t-5.4 ( 5.4 )\nreserve equalization\t5.6\nretail electric price\t19.0\nother\t4.4\n2017 net revenue\t$ 626.8",
        "question": "what percent did net revenue decrease between 2016 and 2017?",
         "answer": "Calculation:\n1. 2016 net revenue: $644.2 million\n2. 2017 net revenue: $626.8 million\n3. Decrease = $644.2 million - $626.8 million = $17.4 million\n4. Percent decrease = ($17.4 million / $644.2 million) * 100 = 2.7%\n\nAnswer: 2.7%"
    },
    {
        "context": "Audi RS5 Coupe: Until a few years ago Audi produced only one RS model at any time.",
        "question": "Given the input sentence, please extract the subject and object containing a certain relation in the sentence according to the following relation types, in the format of \"relation1: word1, word2; relation2: word3, word4\". Relations include: product/material produced; manufacturer; distributed by; industry; position held; original broadcaster; owned by; founded by; distribution format; headquarters location; stock exchange; currency; parent organization; chief executive officer; director/manager; owner of; operator; member of; employer; chairperson; platform; subsidiary; legal form; publisher; developer; brand; business division; location of formation; creator.",
        "answer": "manufacturer: Audi RS5, Audi; brand: Audi RS5, Audi"
    },
]
# Modified prompt template
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
        relevant_examples = [few_shot_examples[i] for i in relevant_indices]
        formatted_examples = format_few_shot_examples(relevant_examples)
        
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

    dataset = Dataset.from_dict({
        "context": [item["context"] for item in train_data],
        "question": [item["question"] for item in train_data],
        "answer": [item["answer"] for item in train_data],
    })

    # Pre-compute relevant examples
    relevant_examples_file = "relevant_examples.json"
    if not os.path.exists(relevant_examples_file):
        relevant_examples_map = precompute_relevant_examples(dataset, few_shot_examples, top_k, context_weight, question_weight)
        save_relevant_examples(relevant_examples_map, relevant_examples_file)
    else:
        relevant_examples_map = load_relevant_examples(relevant_examples_file)

    # Add relevant examples to the dataset
    dataset = dataset.add_column("relevant_examples", [relevant_examples_map.get(str(i), []) for i in range(len(dataset))])
    
    dataset = preprocess_dataset(tokenizer, max_length, seed, dataset, few_shot_examples)
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
    dataset = prepare_dataset(tokenizer, max_length, seed=9, few_shot_examples=few_shot_examples, top_k=1, context_weight=0.5, question_weight=0.5)
    
    # Initialize distributed training
    dist.init_process_group(backend="nccl")
    
    output_dir = "fewshot_similarexamples_llama3"
    train(model, tokenizer, dataset, output_dir)
    
    if dist.get_rank() == 0:
        if os.path.exists(output_dir):
            print("Model file exists, proceeding with loading and merging...")
            try:
                model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
                model = model.merge_and_unload()
                output_merged_dir = "fewshot_similarexamples_llama3/final_merged_checkpoint"
                os.makedirs(output_merged_dir, exist_ok=True)
                model.save_pretrained(output_merged_dir, safe_serialization=True)
                tokenizer.save_pretrained(output_merged_dir)
                print(f"Model merged and saved in {output_merged_dir}")
            except Exception as e:
                print(f"Error during model loading or merging: {e}")
        else:
            print("Model file does not exist. Please check the training and saving steps.")
    dist.destroy_process_group()
