#!pip install --upgrade transformers -> for llama 3.1
#!pip install datasets evaluate rouge_score bert_score

#filtered_test_data with python script

import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_metric
import evaluate
from difflib import SequenceMatcher
import re
import numpy as np

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
#meta-llama/Meta-Llama-3.1-8B
# Load the LLaMA 3 8B model and tokenizer
model_name = "meta-llama/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Load metrics
rouge = load_metric('rouge')
bleu = load_metric('bleu')
meteor = load_metric('meteor')
bertscore = load_metric('bertscore')
google_bleu = evaluate.load("google_bleu")


# Improved prompt template
IMPROVED_FEW_SHOT_PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with examples and an input that provides further context. Learn from the examples and write a response that appropriately completes the request.

### Instruction:
You are an AI assistant specialized in financial analysis. Your task is to generate Python code that answers questions based on given financial data and context. Learn from the examples provided and generate accurate, executable Python code that solves the given problem.

Important: When writing Python code, use actual newline characters to separate lines, not '\n' string literals. Each line of code should be on its own line in your response.

### Examples:
{examples}

### Input:
Context:
{context}

Question: {question}

### Response:
Here's the Python code to answer the question:

```python
"""

# Load examples from JSON file
def load_examples(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

examples = load_examples('script_data.json')

# Function to select most relevant examples
def select_relevant_examples(examples, input_context, input_question, n=3):
    vectorizer = TfidfVectorizer().fit_transform([ex['context'] + ' ' + ex['question'] for ex in examples] + [input_context + ' ' + input_question])
    similarities = cosine_similarity(vectorizer[-1], vectorizer[:-1]).flatten()
    top_indices = similarities.argsort()[-n:][::-1]
    return [examples[i] for i in top_indices]

# Function to format examples for few-shot prompting
def format_examples(examples):
    formatted = ""
    for ex in examples:
        formatted += f"Context:\n{ex['context']}\n\n"
        formatted += f"Question: {ex['question']}\n\n"
        formatted += f"Python Code:\n```python\n{ex['answer']}\n```\n\n"
        formatted += "---\n\n"
    return formatted

def generate_code(context, question, few_shot_examples):
    formatted_examples = format_examples(few_shot_examples)
    prompt = IMPROVED_FEW_SHOT_PROMPT_TEMPLATE.format(
        examples=formatted_examples,
        context=context,
        question=question
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=2000,
            num_return_sequences=1,
            temperature=0.9,
            max_new_tokens=150,
            top_p=0.95,
            do_sample=True
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the code block from the generated text
    code_start = generated_text.rfind("```python")
    code_end = generated_text.rfind("```")
    if code_start != -1 and code_end != -1:
        generated_code = generated_text[code_start+9:code_end].strip()
    else:
        generated_code = ""  # Return empty string if no valid Python code block found

    return generated_code

def execute_code(code, context):
    # Replace '\n' string literals with actual newlines (as a safety measure)
    code = code.replace('\\n', '\n')

    # Create a safe environment for code execution
    safe_globals = {
        'pd': pd,
        'context': context
    }

    try:
        # Use exec to run the code and capture the output
        exec(code, safe_globals)

        # Check if 'result' variable is defined in the executed code
        if 'result' in safe_globals:
            return safe_globals['result'], None

        # If 'result' is not defined, check for any printed output
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        exec(code, safe_globals)

        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        return output.strip() if output else "No output or result variable found", None
    except Exception as e:
        return None, str(e)


def process_financial_query(context, question):
    # Select relevant examples
    selected_examples = select_relevant_examples(examples, context, question)

    # Generate Python code
    generated_code = generate_code(context, question, selected_examples)

    # Execute the generated code
    result, error = execute_code(generated_code, context)

    return {
        "generated_code": generated_code,
        "result": result,
        "error": error
    }

def normalize_text(text):
    # Remove repeated sequences and extra spaces
    text = re.sub(r'(\b\w+\b)(?=.*?\b\1\b)', '', text)
    text = ' '.join(text.split())
    return text

def calculate_accuracy_exact(pred, ref):
    pred_num = extract_numerical_value(pred)
    ref_num = extract_numerical_value(ref)
    #print(f'pred after extraction: {pred_num} \t ref after extraction: {ref_num}')
    if pred_num is not None and ref_num is not None:
      return 1 if pred_num == ref_num else 0 #for num
    else:
      return 1 if pred.strip() == ref.strip() else 0 #for text

def calculate_accuracy_some_leniency(predicted, actual):
    predicted = str(predicted).lower().strip()
    actual = str(actual).lower().strip()

    # Try to extract numerical values
    pred_num = extract_numerical_value(predicted)
    actual_num = extract_numerical_value(actual)

    if pred_num is not None and actual_num is not None:
        # If both are numbers, check if they're close enough
        return int(abs(pred_num - actual_num) < 0.001)
    else:
        # If not both numbers, do a more flexible string comparison
        # Split the strings into words
        pred_words = predicted.split()
        actual_words = actual.split()
        # Calculate the similarity ratio
        similarity = SequenceMatcher(None, pred_words, actual_words).ratio()
        key_parts_match = all(part in predicted for part in actual.split(':')[-1].split(','))
        # Return 1 if similarity is high, 0 otherwise
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

    # Compute ROUGE
    rouge_result = rouge.compute(predictions=predictions, references=references)
    rouge_score = rouge_result['rougeL'].mid.fmeasure

    # Compute BLEU
    bleu_result = bleu.compute(predictions=tokenized_predictions, references=tokenized_references)
    bleu_score = bleu_result['bleu']

    # Compute METEOR
    meteor_result = meteor.compute(predictions=predictions, references=references)
    meteor_score = meteor_result['meteor']

    # Compute BERTScore
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

def process_test_samples(file_path, num_samples,seed=None):
    if seed is not None:
        random.seed(seed)
    with open(file_path, 'r') as f:
        all_test_samples = json.load(f)

    # Randomly select 1000 samples (or less if there are fewer than 1000 samples)
    total_samples = len(all_test_samples)
    num_samples = min(num_samples, total_samples)
    selected_samples = random.sample(all_test_samples, num_samples)

    filtered_samples = []
    filtered_count = 0
    error_count = 0
    leniency_accuracy_scores = []
    exact_accuracy_scores = []
    all_predictions = []
    all_references = []

    for sample in tqdm(selected_samples, desc="Processing samples", unit="sample"):
        context = sample['context']
        question = sample['question']
        reference_value = str(sample['answer'])
        result = process_financial_query(context, question)

        if result['generated_code']:
            sample['answer'] = result['generated_code'].replace('\n', '\\n')
            filtered_samples.append(sample)

            if result['error']:
                error_count += 1
            else:
                predicted_value = str(result['result'])
                #print(f'context: {context} \n question: {question} \n predicted: {predicted_value} \n reference: {reference_value}')
                all_predictions.append(predicted_value)
                all_references.append(reference_value)
                leniency_accuracy_scores.append(calculate_accuracy_some_leniency(predicted_value, reference_value))
                exact_accuracy_scores.append(calculate_accuracy_exact(predicted_value, reference_value))
        else:
            filtered_count += 1

    # Save filtered samples to a new file
    output_file = 'filtered_test_data_1000.json'
    with open(output_file, 'w') as f:
        json.dump(filtered_samples, f, indent=2)

    # Calculate metrics
    metric_results = evaluate_metrics(all_predictions, all_references)

    # Print results
    print(f"Processing completed. {filtered_count} samples were filtered out.")
    print(f"Total samples selected: {num_samples}")
    print(f"Remaining samples: {len(filtered_samples)}")
    print(f"Percentage of samples with execution errors: {(error_count / len(filtered_samples)) * 100:.2f}%")
    print(f"Average Permissive Accuracy: {sum(leniency_accuracy_scores) / len(leniency_accuracy_scores):.2f}")
    print(f"Average Exact Accuracy: {sum(exact_accuracy_scores) / len(exact_accuracy_scores):.2f}")
    print(f"ROUGE Score: {metric_results['rouge']:.2f}")
    print(f"BLEU Score: {metric_results['bleu']:.2f}")
    print(f"METEOR Score: {metric_results['meteor']:.2f}")
    print(f"BERTScore Precision: {metric_results['bertscore_precision']:.3f}")
    print(f"BERTScore Recall: {metric_results['bertscore_recall']:.3f}")
    print(f"BERTScore F1: {metric_results['bertscore_f1']:.3f}")
    print(f"GLEU Score: {metric_results['googlebleuscore']:.2f}")
    print(f"Results have been saved to {output_file}")

# Process test samples
process_test_samples('tat_qa_dataset.json', num_samples=1000,seed=9)
