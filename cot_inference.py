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

    file_path = "merged_test_data.json"
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

cot_prompt = """You are a financial expert assistant. Answer the following question based on the given context. It is CRUCIAL that you use step-by-step reasoning and provide detailed numerical calculations where applicable. Break down your answer into clear, numbered steps. If asked to extract subject and object in relation, return only the single most relevant and applicable extraction. Use '\n' for line breaks in your response.

Context: {context}
Question: {question}
Answer (including calculation steps and final answer, or single most relevant relation extraction, use '\n' for line breaks):"""

def create_prompt_format(sample):
    text = cot_prompt.format(
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

def inference_and_scoring(samples, model, tokenizer, file_handle, config_name):
    leniency_accuracy_scores = []
    exact_accuracy_scores = []
    all_predictions = []
    all_references = []

    for sample in tqdm(samples, desc="Processing samples", unit="sample"):
        formatted_sample = create_prompt_format(sample)

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
    seed_value = 7
    random.seed(seed_value)
    test_set = prepare_dataset_test(76)
    test_samples = select_random_samples(test_set, 20, seed=seed_value)
    
    model_path = "/home/araj/cot_llama3/final_merged_checkpoint"
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    with open("evaluation_results_cot.txt", "w") as file:
        config_name = "CoT-FineTuned"
        print(f"Running configuration: {config_name}")
        inference_and_scoring(test_set, model, tokenizer, file, config_name)
