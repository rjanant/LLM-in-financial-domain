# import torch
# import re
# from difflib import SequenceMatcher
# from tqdm import tqdm
# from datasets import load_dataset, load_metric, Dataset
# import json
# import random
# import os
# import warnings
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# import evaluate
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP

# warnings.filterwarnings('ignore')

# def setup():
#     # initialize the process group
#     dist.init_process_group("nccl")

# def cleanup():
#     dist.destroy_process_group()

# def prepare_dataset_test(seed):
#     # Set the random seed
#     random.seed(seed)

#     file_path = "finqa_data.json"
#     with open(file_path, "r") as f:
#         dataset = json.load(f)
#     random.seed(seed)
#     sample_size = len(dataset) // 5
#     sampled_dataset = random.sample(dataset, sample_size)

#     data_dict_test = {
#         "context": [item["context"] for item in sampled_dataset],
#         "question": [item["question"] for item in sampled_dataset],
#         "answer": [item["answer"] for item in sampled_dataset]
#     }
#     test_dataset = Dataset.from_dict(data_dict_test)

#     return test_dataset

# def select_random_samples(dataset, num_samples, seed=None):
#     if seed is not None:
#         random.seed(seed)
#     random_indices = random.sample(range(0, len(dataset)), num_samples)
#     random_samples = dataset.select(random_indices)
#     return random_samples

# # Load the embedding model
# embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# def get_embeddings(text):
#     with torch.no_grad():
#         embedding = embedding_model.encode(text, convert_to_tensor=True)
#     return embedding.cpu().numpy()

# # Load few-shot examples
# with open("few_shot_examples.json", "r") as f:
#     few_shot_examples = json.load(f)

# few_shot_prompt = """You are a financial expert assistant. Answer the following question based on the given context. It is CRUCIAL that you use step-by-step reasoning and provide detailed numerical calculations where applicable. Break down your answer into clear, numbered steps. If asked to extract subject and object in relation, return only the single most relevant and applicable extraction. Use '\n' for line breaks in your response.

# Examples:
# {examples}

# Now, please answer the following question using the same step-by-step approach as demonstrated in the Examples:

# Context: {context}
# Question: {question}
# Answer (including calculation steps and final answer, or single most relevant relation extraction, use '\n' for line breaks):"""

# def format_few_shot_examples(examples):
#     formatted = ""
#     for i, example in enumerate(examples, 1):
#         formatted += f"Example {i}:\n"
#         formatted += f"Context: {example['context']}\n"
#         formatted += f"Question: {example['question']}\n"
#         formatted += f"Answer: {example['answer']}\n\n"
#     return formatted

# def find_relevant_examples(sample, few_shot_examples, top_k, context_weight=0.7, question_weight=0.5):
#     sample_context_embedding = get_embeddings(sample['context'])
#     sample_question_embedding = get_embeddings(sample['question'])
    
#     similarities = []
#     for example in few_shot_examples:
#         example_context_embedding = get_embeddings(example['context'])
#         example_question_embedding = get_embeddings(example['question'])
        
#         context_similarity = cosine_similarity([sample_context_embedding], [example_context_embedding])[0][0]
#         question_similarity = cosine_similarity([sample_question_embedding], [example_question_embedding])[0][0]
        
#         combined_similarity = (context_weight * context_similarity) + (question_weight * question_similarity)
#         similarities.append(combined_similarity)
    
#     top_indices = np.argsort(similarities)[-top_k:][::-1]
#     return [few_shot_examples[i] for i in top_indices]

# def create_prompt_format(sample, example_type):
#     if example_type == 'zero_shot':
#         examples = []
#     elif example_type == 'one_shot':
#         examples = find_relevant_examples(sample, few_shot_examples, top_k=1)
#     elif example_type == 'two_shot':
#         examples = find_relevant_examples(sample, few_shot_examples, top_k=2)
#     elif example_type == 'random_shot':
#         examples = random.sample(few_shot_examples, 1)
#     elif example_type == 'mix_shot':
#         relevant_examples = find_relevant_examples(sample, few_shot_examples, top_k=1)
#         random_examples = random.sample(few_shot_examples, 1)
#         examples = relevant_examples + random_examples
#     else:
#         examples = []

#     formatted_examples = format_few_shot_examples(examples)

#     text = few_shot_prompt.format(
#         examples=formatted_examples,
#         context=sample["context"],
#         question=sample["question"]
#     )
#     return {"text": text}

# # Load metrics
# rouge = load_metric('rouge')
# bleu = load_metric('bleu')
# meteor = load_metric('meteor')
# bertscore = load_metric('bertscore')
# google_bleu = evaluate.load("google_bleu")

# def preprocess_output(output):
#     answer_pattern = r"Answer \(including calculation steps and final answer, or single most relevant relation extraction, use '\n' for line breaks\):(.*?)(?:\n\nContext:|$)"
#     match = re.search(answer_pattern, output, re.DOTALL)
#     if match:
#         answer = match.group(1).strip()
#         lines = [line.strip() for line in answer.split('\n') if line.strip()]
#         return lines[0] if lines else ""
#     return ""

# def calculate_accuracy_exact(pred, ref):
#     return 1 if pred.strip() == ref.strip() else 0

# def calculate_accuracy_some_leniency(predicted, actual):
#     predicted = str(predicted).lower().strip()
#     actual = str(actual).lower().strip()
    
#     pred_num = extract_numerical_value(predicted)
#     actual_num = extract_numerical_value(actual)
    
#     if pred_num is not None and actual_num is not None:
#         return int(abs(pred_num - actual_num) < 0.001)
#     else:
#         pred_words = predicted.split()
#         actual_words = actual.split()
        
#         similarity = SequenceMatcher(None, pred_words, actual_words).ratio()
        
#         key_parts_match = all(part in predicted for part in actual.split(':')[-1].split(','))
        
#         return int(similarity > 0.9 or key_parts_match)

# def extract_numerical_value(text):
#     match = re.search(r'-?\d+(\.\d+)?', text)
#     if match:
#         value = match.group()
#         return float(value) if '.' in value else int(value)
#     return None

# def evaluate_metrics(predictions, references):
#     tokenized_predictions = [pred.split() for pred in predictions]
#     tokenized_references = [[ref.split()] for ref in references]
    
#     rouge_result = rouge.compute(predictions=predictions, references=references)
#     rouge_score = rouge_result['rougeL'].mid.fmeasure

#     bleu_result = bleu.compute(predictions=tokenized_predictions, references=tokenized_references)
#     bleu_score = bleu_result['bleu']

#     meteor_result = meteor.compute(predictions=predictions, references=references)
#     meteor_score = meteor_result['meteor']

#     bertscore_results = bertscore.compute(predictions=predictions, references=references, lang="en")
#     bertscore_precision = np.mean(bertscore_results['precision'])
#     bertscore_recall = np.mean(bertscore_results['recall'])
#     bertscore_f1 = np.mean(bertscore_results['f1'])

#     google_bleu_result = google_bleu.compute(predictions=predictions, references=references)
#     google_bleu_score = np.mean(google_bleu_result['google_bleu'])

#     results = {
#         "rouge": rouge_score,
#         "bleu": bleu_score,
#         "meteor": meteor_score,
#         "bertscore_precision": bertscore_precision,
#         "bertscore_recall": bertscore_recall,
#         "bertscore_f1": bertscore_f1,
#         "google_bleuscore": google_bleu_score
#     }
#     return results

# example_type_names = [
#     'Zero Shot',      # No examples provided
#     'One Shot',       # One relevant example
#     'Two Shot',       # Two relevant examples
#     'Random Shot',    # One random example
#     'Mix Shot'        # Combination of one relevant and one random example
# ]

# def inference_and_scoring(samples, model, tokenizer, weights, config_name, rank, world_size,example_type_choices):
#     # Distribute samples across GPUs
#     per_gpu_samples = len(samples) // world_size
#     start_idx = rank * per_gpu_samples
#     end_idx = start_idx + per_gpu_samples if rank != world_size - 1 else len(samples)
#     gpu_samples = samples[start_idx:end_idx]

#     leniency_accuracy_scores = []
#     exact_accuracy_scores = []
#     all_predictions = []
#     all_references = []
#     example_type_counts = {choice: 0 for choice in example_type_choices}

#     for sample in tqdm(gpu_samples, desc=f"Processing samples on GPU {rank}", unit="sample"):
#         example_type = random.choices(example_type_choices, weights=weights, k=1)[0]
#         example_type_counts[example_type] += 1

#         formatted_sample = create_prompt_format(sample, example_type)

#         inputs = tokenizer(
#             formatted_sample['text'], return_tensors="pt", truncation=True, max_length=8096
#         ).to(rank)

#         outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=False, num_beams=1)
#         predicted_value = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         predicted_value = preprocess_output(predicted_value)

#         all_predictions.append(predicted_value)
#         all_references.append(sample["answer"])
#         leniency_accuracy_scores.append(calculate_accuracy_some_leniency(predicted_value, sample["answer"]))
#         exact_accuracy_scores.append(calculate_accuracy_exact(predicted_value, sample["answer"]))

#     # Gather results from all GPUs
#     all_leniency_scores = [None] * world_size
#     all_exact_scores = [None] * world_size
#     all_predictions_gathered = [None] * world_size
#     all_references_gathered = [None] * world_size
#     all_example_type_counts = [None] * world_size

#     dist.all_gather_object(all_leniency_scores, leniency_accuracy_scores)
#     dist.all_gather_object(all_exact_scores, exact_accuracy_scores)
#     dist.all_gather_object(all_predictions_gathered, all_predictions)
#     dist.all_gather_object(all_references_gathered, all_references)
#     dist.all_gather_object(all_example_type_counts, example_type_counts)

#     if rank == 0:
#         # Combine results
#         leniency_accuracy_scores = [score for scores in all_leniency_scores for score in scores]
#         exact_accuracy_scores = [score for scores in all_exact_scores for score in scores]
#         all_predictions = [pred for preds in all_predictions_gathered for pred in preds]
#         all_references = [ref for refs in all_references_gathered for ref in refs]
        
#         for counts in all_example_type_counts:
#             for example_type, count in counts.items():
#                 example_type_counts[example_type] += count

#         metric_results = evaluate_metrics(all_predictions, all_references)

#         # Write results to file (only on rank 0)
#         with open(f"evaluation_results_finqa_{config_name}.txt", "w") as file:
#             file.write(f'----- Results for {config_name} -----\n')
#             file.write("Example type distribution:\n")
#             for example_type, count in example_type_counts.items():
#                 file.write(f"{example_type}: {count}\n")
#             file.write(f"Average Permissive Accuracy: {sum(leniency_accuracy_scores) / len(leniency_accuracy_scores):.2f}\n")
#             file.write(f"Average Exact Accuracy: {sum(exact_accuracy_scores) / len(exact_accuracy_scores):.2f}\n")
#             file.write(f"ROUGE Score: {metric_results['rouge']:.2f}\n")
#             file.write(f"BLEU Score: {metric_results['bleu']:.2f}\n")
#             file.write(f"METEOR Score: {metric_results['meteor']:.2f}\n")
#             file.write(f"BERTScore Precision: {metric_results['bertscore_precision']:.3f}\n")
#             file.write(f"BERTScore Recall: {metric_results['bertscore_recall']:.3f}\n")
#             file.write(f"BERTScore F1: {metric_results['bertscore_f1']:.3f}\n")
#             file.write(f"GLEU Score: {metric_results['google_bleuscore']:.2f}\n")
#             file.write('\n')

#         print(f'----- Results for {config_name} -----')
#         print("Example type distribution:")
#         for example_type, count in example_type_counts.items():
#             print(f"{example_type}: {count}")
#         print(f"Average Permissive Accuracy : {sum(leniency_accuracy_scores) / len(leniency_accuracy_scores):.2f}")
#         print(f"Average Exact Accuracy: {sum(exact_accuracy_scores) / len(exact_accuracy_scores):.2f}")
#         print(f"ROUGE Score: {metric_results['rouge']:.2f}")
#         print(f"BLEU Score: {metric_results['bleu']:.2f}")
#         print(f"METEOR Score: {metric_results['meteor']:.2f}")
#         print(f"BERTScore Precision: {metric_results['bertscore_precision']:.3f}")
#         print(f"BERTScore Recall: {metric_results['bertscore_recall']:.3f}")
#         print(f"BERTScore F1: {metric_results['bertscore_f1']:.3f}")
#         print(f"GLEU Score: {metric_results['google_bleuscore']:.2f}")

#     dist.barrier()

# def main():
#     local_rank = int(os.environ["LOCAL_RANK"])
#     world_size = int(os.environ["WORLD_SIZE"])
    
#     setup()
    
#     model_path = "/home/araj/0to2fewshot_examples_llama3/final_merged_checkpoint"
#     model = AutoModelForCausalLM.from_pretrained(model_path, device_map={"": local_rank}, torch_dtype=torch.bfloat16)
#     model = DDP(model, device_ids=[local_rank])
#     tokenizer = AutoTokenizer.from_pretrained(model_path)

#     seed_value = 7
#     random.seed(seed_value)
#     test_set = prepare_dataset_test(76)

#     example_type_choices = [
#         'zero_shot',      # No examples provided
#         'one_shot',       # One relevant example
#         'two_shot',       # Two relevant examples
#         'random_shot',    # One random example
#         'mix_shot'        # Combination of one relevant and one random example
#     ]

#     weight_configurations = [
#         [1.00, 0.00, 0.00, 0.00, 0.00],  # Zero-shot
#         [0.00, 1.00, 0.00, 0.00, 0.00],  # One-shot
#         [0.00, 0.00, 1.00, 0.00, 0.00],  # Two-shot
#         [0.00, 0.00, 0.00, 1.00, 0.00],  # Random-shot
#         [0.00, 0.00, 0.00, 0.00, 1.00]   # Mix-shot
#     ]

#     for i, weights in enumerate(weight_configurations):
#         config_name = example_type_names[i]
#         if local_rank == 0:
#             print(f"Running configuration: {config_name}")
#         inference_and_scoring(test_set, model.module, tokenizer, weights, config_name, local_rank, world_size,example_type_choices)

#     cleanup()

# if __name__ == "__main__":
#     main()

import torch
import re
from difflib import SequenceMatcher
from tqdm import tqdm
from datasets import load_metric, Dataset
import json
import random
import os
import warnings
import json
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings('ignore')

def prepare_dataset_test(seed):
    # Set the random seed
    random.seed(seed)

    file_path = "finqa_data.json"
    with open(file_path, "r") as f:
        dataset = json.load(f)
    random.seed(seed)
    sample_size = len(dataset) // 5
    sampled_dataset = random.sample(dataset, sample_size)

    data_dict_test = {
        "context": [item["context"] for item in sampled_dataset],
        "question": [item["question"] for item in sampled_dataset],
        "answer": [item["answer"] for item in sampled_dataset]
    }
    test_dataset = Dataset.from_dict(data_dict_test)

    return test_dataset

#test_set = prepare_dataset_test(76)
def select_random_samples(dataset, num_samples, seed=None):
    if seed is not None:
        random.seed(seed)
    random_indices = random.sample(range(0, len(dataset)), num_samples)
    random_samples = dataset.select(random_indices)
    return random_samples

# Load the embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embeddings(text):
    with torch.no_grad():
        embedding = embedding_model.encode(text, convert_to_tensor=True)
    return embedding.cpu().numpy()

# Load few-shot examples
with open("few_shot_examples.json", "r") as f:
    few_shot_examples = json.load(f)

few_shot_prompt = """You are a financial expert assistant. Answer the following question based on the given context. It is CRUCIAL that you use step-by-step reasoning and provide detailed numerical calculations where applicable. Break down your answer into clear, numbered steps. If asked to extract subject and object in relation, return only the single most relevant and applicable extraction. Use '\n' for line breaks in your response.

Examples:
{examples}

Now, please answer the following question using the same step-by-step approach as demonstrated in the Examples:

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

def find_relevant_examples(sample, few_shot_examples, top_k, context_weight=0.7, question_weight=0.5):
    sample_context_embedding = get_embeddings(sample['context'])
    sample_question_embedding = get_embeddings(sample['question'])
    
    similarities = []
    for example in few_shot_examples:
        example_context_embedding = get_embeddings(example['context'])
        example_question_embedding = get_embeddings(example['question'])
        
        context_similarity = cosine_similarity([sample_context_embedding], [example_context_embedding])[0][0]
        question_similarity = cosine_similarity([sample_question_embedding], [example_question_embedding])[0][0]
        
        combined_similarity = (context_weight * context_similarity) + (question_weight * question_similarity)
        similarities.append(combined_similarity)
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [few_shot_examples[i] for i in top_indices]

def create_prompt_format(sample, example_type):
    if example_type == 'zero_shot':
        examples = []
    elif example_type == 'one_shot':
        examples = find_relevant_examples(sample, few_shot_examples, top_k=1)
    elif example_type == 'two_shot':
        examples = find_relevant_examples(sample, few_shot_examples, top_k=2)
    elif example_type == 'random_shot':
        examples = random.sample(few_shot_examples, 1)
    elif example_type == 'mix_shot':
        relevant_examples = find_relevant_examples(sample, few_shot_examples, top_k=1)
        random_examples = random.sample(few_shot_examples, 1)
        examples = relevant_examples + random_examples
    else:
        examples = []

    formatted_examples = format_few_shot_examples(examples)

    text = few_shot_prompt.format(
        examples=formatted_examples,
        context=sample["context"],
        question=sample["question"]
    )
    return {"text": text}


# Load metrics
rouge = load_metric('rouge')
bleu = load_metric('bleu')
meteor = load_metric('meteor')
bertscore = load_metric('bertscore')
google_bleu = evaluate.load("google_bleu")

def preprocess_output(output):
    answer_pattern = r"Answer \(including calculation steps and final answer, or single most relevant relation extraction, use '\n' for line breaks\):(.*?)(?:\n\nContext:|$)"
    match = re.search(answer_pattern, output, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        lines = [line.strip() for line in answer.split('\n') if line.strip()]
        return lines[0] if lines else ""
    return ""

def calculate_accuracy_exact(pred, ref):
    return 1 if pred.strip() == ref.strip() else 0

def calculate_accuracy_some_leniency(predicted, actual):
    predicted = str(predicted).lower().strip()
    actual = str(actual).lower().strip()
    
    pred_num = extract_numerical_value(predicted)
    actual_num = extract_numerical_value(actual)
    
    if pred_num is not None and actual_num is not None:
        return int(abs(pred_num - actual_num) < 0.001)
    else:
        pred_words = predicted.split()
        actual_words = actual.split()
        
        similarity = SequenceMatcher(None, pred_words, actual_words).ratio()
        
        key_parts_match = all(part in predicted for part in actual.split(':')[-1].split(','))
        
        return int(similarity > 0.9 or key_parts_match)

def extract_numerical_value(text):
    match = re.search(r'-?\d+(\.\d+)?', text)
    if match:
        value = match.group()
        return float(value) if '.' in value else int(value)
    return None

def evaluate_metrics(predictions, references):
    tokenized_predictions = [pred.split() for pred in predictions]
    tokenized_references = [[ref.split()] for ref in references]
    
    rouge_result = rouge.compute(predictions=predictions, references=references)
    rouge_score = rouge_result['rougeL'].mid.fmeasure

    bleu_result = bleu.compute(predictions=tokenized_predictions, references=tokenized_references)
    bleu_score = bleu_result['bleu']

    meteor_result = meteor.compute(predictions=predictions, references=references)
    meteor_score = meteor_result['meteor']

    bertscore_results = bertscore.compute(predictions=predictions, references=references, lang="en")
    bertscore_precision = np.mean(bertscore_results['precision'])
    bertscore_recall = np.mean(bertscore_results['recall'])
    bertscore_f1 = np.mean(bertscore_results['f1'])

    google_bleu_result = google_bleu.compute(predictions=predictions, references=references)
    google_bleu_score = np.mean(google_bleu_result['google_bleu'])

    results = {
        "rouge": rouge_score,
        "bleu": bleu_score,
        "meteor": meteor_score,
        "bertscore_precision": bertscore_precision,
        "bertscore_recall": bertscore_recall,
        "bertscore_f1": bertscore_f1,
        "google_bleuscore": google_bleu_score
    }
    return results

example_type_names = [
    'Zero Shot',      # No examples provided
    'One Shot',       # One relevant example
    'Two Shot',       # Two relevant examples
    'Random Shot',    # One random example
    'Mix Shot'        # Combination of one relevant and one random example
]

def inference_and_scoring(samples, model, tokenizer, weights,file_handle, config_name):
    leniency_accuracy_scores = []
    exact_accuracy_scores = []
    all_predictions = []
    all_references = []
    example_type_counts = {choice: 0 for choice in example_type_choices}

    for sample in tqdm(samples, desc="Processing samples", unit="sample"):
        # Randomly choose an example type for this sample
        example_type = random.choices(example_type_choices, weights=weights, k=1)[0]
        example_type_counts[example_type] += 1

        formatted_sample = create_prompt_format(sample, example_type)

        inputs = tokenizer(
            formatted_sample['text'], return_tensors="pt", truncation=True, max_length=8096
        )

        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=False, num_beams=1)
        predicted_value = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_value = preprocess_output(predicted_value)

        all_predictions.append(predicted_value)
        all_references.append(sample["answer"])
        leniency_accuracy_scores.append(calculate_accuracy_some_leniency(predicted_value, sample["answer"]))
        exact_accuracy_scores.append(calculate_accuracy_exact(predicted_value, sample["answer"]))

    metric_results = evaluate_metrics(all_predictions, all_references)

    # Write results to file
    file_handle.write(f'----- Results for {config_name} -----\n')
    file_handle.write("Example type distribution:\n")
    for example_type, count in example_type_counts.items():
        file_handle.write(f"{example_type}: {count}\n")
    file_handle.write(f"Average Permissive Accuracy: {sum(leniency_accuracy_scores) / len(leniency_accuracy_scores):.2f}\n")
    file_handle.write(f"Average Exact Accuracy: {sum(exact_accuracy_scores) / len(exact_accuracy_scores):.2f}\n")
    file_handle.write(f"ROUGE Score: {metric_results['rouge']:.2f}\n")
    file_handle.write(f"BLEU Score: {metric_results['bleu']:.2f}\n")
    file_handle.write(f"METEOR Score: {metric_results['meteor']:.2f}\n")
    file_handle.write(f"BERTScore Precision: {metric_results['bertscore_precision']:.3f}\n")
    file_handle.write(f"BERTScore Recall: {metric_results['bertscore_recall']:.3f}\n")
    file_handle.write(f"BERTScore F1: {metric_results['bertscore_f1']:.3f}\n")
    file_handle.write(f"GLEU Score: {metric_results['google_bleuscore']:.2f}\n")
    file_handle.write('\n')

    print('----- Results -----')
    print("Example type distribution:")
    for example_type, count in example_type_counts.items():
        print(f"{example_type}: {count}")
    print(f"Average Permissive Accuracy : {sum(leniency_accuracy_scores) / len(leniency_accuracy_scores):.2f}")
    print(f"Average Exact Accuracy: {sum(exact_accuracy_scores) / len(exact_accuracy_scores):.2f}")
    print(f"ROUGE Score: {metric_results['rouge']:.2f}")
    print(f"BLEU Score: {metric_results['bleu']:.2f}")
    print(f"METEOR Score: {metric_results['meteor']:.2f}")
    print(f"BERTScore Precision: {metric_results['bertscore_precision']:.3f}")
    print(f"BERTScore Recall: {metric_results['bertscore_recall']:.3f}")
    print(f"BERTScore F1: {metric_results['bertscore_f1']:.3f}")
    print(f"GLEU Score: {metric_results['google_bleuscore']:.2f}")


if __name__ == "__main__":
        example_type_choices = [
        'zero_shot',      # No examples provided
        'one_shot',       # One relevant example
        'two_shot',       # Two relevant examples
        'random_shot',    # One random example
        'mix_shot'        # Combination of one relevant and one random example
    ]
        weight_configurations = [
        [1.00, 0.00, 0.00, 0.00, 0.00],  # Zero-shot
        [0.00, 1.00, 0.00, 0.00, 0.00],  # One-shot
        [0.00, 0.00, 1.00, 0.00, 0.00],  # Two-shot
        [0.00, 0.00, 0.00, 1.00, 0.00],  # Random-shot
        [0.00, 0.00, 0.00, 0.00, 1.00]   # Mix-shot
    ]

        seed_value = 7
        random.seed(seed_value)
        test_set = prepare_dataset_test(76)
        test_samples = select_random_samples(test_set, 20, seed=seed_value)
        
        model_path = "/home/araj/0to2fewshot_examples_llama3/final_merged_checkpoint"
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # for i, weights in enumerate(weight_configurations):
        #     print(f"Running configuration {i}")
        #     inference_and_scoring(test_samples, model, tokenizer, weights)
        with open("evaluation_results_finqa.txt", "w") as file:
            for i, weights in enumerate(weight_configurations):
                config_name = example_type_names[i]  
                print(f"Running configuration: {config_name}")
                inference_and_scoring(test_set, model, tokenizer, weights, file, config_name)
