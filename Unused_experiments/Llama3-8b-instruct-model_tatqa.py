import os
import torch
import re
from decimal import Decimal, InvalidOperation
import numpy as np
from datasets import load_metric, Dataset
from difflib import SequenceMatcher
from tqdm import tqdm  # Use tqdm instead of tqdm.notebook for script usage
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
import random
import json
import warnings

#meta-llama/Meta-Llama-3-8B-instruct base on tat-qa
#os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_merged_dir = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(output_merged_dir, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(output_merged_dir)

warnings.filterwarnings('ignore')

prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately answers the question. Provide ONLY the answer without any explanation or additional text.

### Instruction:
{instruction}

### Question:
{question}

### Response:
"""

def create_prompt_format_nw(sample):
    instruction = sample["context"]
    input_ = sample["question"]
    text = prompt_template.format(instruction=instruction, question=input_)
    return {"text": text}

# Load metrics
rouge = load_metric('rouge')
bleu = load_metric('bleu')
meteor = load_metric('meteor')
bertscore = load_metric('bertscore')
google_bleu = evaluate.load("google_bleu")

def preprocess_output(output):
    response_pattern = r"### Response:\s*(.*?)(?:\n\n###|$)"
    match = re.search(response_pattern, output, re.DOTALL)
    if match:
        response = match.group(1).strip()
        # Remove any explanations or additional text after the first sentence
        first_sentence = re.split(r'(?<=[.!?])\s', response)[0]
        return first_sentence
    return ""

def extract_numerical_value(text):
    match = re.search(r'-?\d+(\.\d+)?', text)
    if match:
        value = match.group()
        return float(value) if '.' in value else int(value)
    return None

def calculate_accuracy_exact(pred, ref):
    pred_num = extract_numerical_value(pred)
    ref_num = extract_numerical_value(ref)
    if pred_num is not None and ref_num is not None:
        return 1 if pred_num == ref_num else 0
    else:
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
        "googlebleuscore": google_bleu_score
    }
    return results

def inference_and_scoring(samples, model, tokenizer):
    leniency_accuracy_scores = []
    exact_accuracy_scores = []
    all_predictions = []
    all_references = []

    for sample in tqdm(samples, desc="Processing samples", unit="sample"):
        formatted_sample = create_prompt_format_nw(sample)
        inputs = tokenizer(
            formatted_sample['text'], return_tensors="pt", truncation=True, max_length=4096
        )

        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7, do_sample=False, num_beams=1)
        predicted_value = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_value = preprocess_output(predicted_value)
        #print(f'context: {sample["context"][:100]} {"..." if len(sample["context"]) > 100 else ""} \n question: {sample["question"]} \n generated output: {predicted_value} \n reference: {sample["answer"]} \n Accuracy: {calculate_accuracy_some_leniency(predicted_value, sample["answer"])} \n {"-" * 50}')
        all_predictions.append(predicted_value)
        all_references.append(sample["answer"])
        leniency_accuracy_scores.append(calculate_accuracy_some_leniency(predicted_value, sample["answer"]))
        exact_accuracy_scores.append(calculate_accuracy_exact(predicted_value, sample["answer"]))

    metric_results = evaluate_metrics(all_predictions, all_references)
    print(f'Results for the Llama3-8b-instruct  model {"-" * 50}')
    print(f"Average Permissive Accuracy : {sum(leniency_accuracy_scores) / len(leniency_accuracy_scores):.2f}")
    print(f"Average Exact Accuracy: {sum(exact_accuracy_scores) / len(exact_accuracy_scores):.2f}")
    print(f"ROUGE Score: {metric_results['rouge']:.2f}")
    print(f"BLEU Score: {metric_results['bleu']:.2f}")
    print(f"METEOR Score: {metric_results['meteor']:.2f}")
    print(f"BERTScore Precision: {metric_results['bertscore_precision']:.3f}")
    print(f"BERTScore Recall: {metric_results['bertscore_recall']:.3f}")
    print(f"BERTScore F1: {metric_results['bertscore_f1']:.3f}")
    print(f"GLEU Score: {metric_results['googlebleuscore']:.2f}")


def select_random_samples(file_path, num_samples, seed=None):
    if seed is not None:
        random.seed(seed)
    
    with open(file_path, 'r') as f:
        all_samples = json.load(f)
    
    total_samples = len(all_samples)
    num_samples = min(num_samples, total_samples)
    selected_samples = random.sample(all_samples, num_samples)
    
    return selected_samples

if __name__ == "__main__":
    seed_value = 9
    file_path = 'tatqa_dataset.json'
    test_samples = select_random_samples(file_path, 1000, seed=seed_value)
    inference_and_scoring(test_samples, model, tokenizer)
