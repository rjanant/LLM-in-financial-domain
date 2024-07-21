#inference - 0,1,2 shot prompt
import json
import random
from datasets import Dataset
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch,os
import torch
import re
from difflib import SequenceMatcher
from tqdm.notebook import tqdm
from datasets import load_metric, Dataset
import warnings
#from sklearn.metrics import precision_recall_fscore_support
import json
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import evaluate

warnings.filterwarnings('ignore')
def prepare_dataset_test(seed):
    # Set the random seed
    random.seed(seed)

    file_path = "merged_test_data.json"
    with open(file_path, "r") as f:
        dataset = json.load(f)
    
    # # Sort the combined_data to ensure consistent ordering
    # combined_data.sort(key=lambda x: json.dumps(x, sort_keys=True))
    random.seed(seed)
    sample_size = len(dataset) // 5
    sampled_dataset = random.sample(dataset, sample_size)

    # train_size = int(0.9 * len(sampled_data))
    # test_data = sampled_data[train_size:]
    

    data_dict_test = {
        "context": [item["context"] for item in sampled_dataset],
        "question": [item["question"] for item in sampled_dataset],
        "answer": [item["answer"] for item in sampled_dataset]
    }
    test_dataset = Dataset.from_dict(data_dict_test)

    return test_dataset

test_set = prepare_dataset_test(76)
# print(test_set) #to check if the test set is same as the one used for other experiments.
# print(test_set[:1])
# print(test_set[7])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_merged_dir = "0to5fewshot_examples_llama3/final_merged_checkpoint"
model = AutoModelForCausalLM.from_pretrained(output_merged_dir, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(output_merged_dir)

# Load the embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embeddings(text):
    with torch.no_grad():
        embedding = embedding_model.encode(text, convert_to_tensor=True)
    return embedding.cpu().numpy()

# Load few-shot examples
with open("few_shot_examples.json", "r") as f:
    few_shot_examples = json.load(f)

# Modified prompt template
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

def find_relevant_examples(sample, few_shot_examples, top_k, context_weight=0.5, question_weight=0.5):
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

def create_prompt_format(sample, num_examples=0):
    if num_examples > 0:
        relevant_examples = find_relevant_examples(sample, few_shot_examples, top_k=num_examples)
        formatted_examples = format_few_shot_examples(relevant_examples)
    else:
        formatted_examples = ""
    
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
    # Extract the final answer from the model's output
    answer_pattern = r"Answer \(including calculation steps and final answer, or single most relevant relation extraction, use '\n' for line breaks\):(.*?)(?:\n\nContext:|$)"
    match = re.search(answer_pattern, output, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        #print('processed answer after removing context & input:', answer)
        # If the answer contains multiple lines, take the first non-empty line
        lines = [line.strip() for line in answer.split('\n') if line.strip()]
        #print('returning first non emptyline: ',lines[0])
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
        "googlebleuscore":google_bleu_score
    }
    return results

def inference_and_scoring(samples, model, tokenizer, num_examples=0):
    leniency_accuracy_scores = []
    exact_accuracy_scores = []
    all_predictions = []
    all_references = []

    # Print the learning mode based on num_examples
    if num_examples == 0:
        print("Running in Zero-shot mode")
    elif num_examples == 1:
        print("Running in One-shot mode")
    elif num_examples == 2:
        print("Running in Two-shot mode")
    else:
        print(f"Running with {num_examples} examples")

    for sample in tqdm(samples, desc="Processing samples", unit="sample"):
        formatted_sample = create_prompt_format(sample, num_examples)
        inputs = tokenizer(
            formatted_sample['text'], return_tensors="pt", truncation=True, max_length=4096
        ).to("cuda")

        outputs = model.generate(**inputs, max_new_tokens=100, temperature=1.0, do_sample=False, num_beams=1)
        predicted_value = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_value = preprocess_output(predicted_value)
        # print(f'prediction: {predicted_value} \n reference: {sample["answer"]}')
        all_predictions.append(predicted_value)
        all_references.append(sample["answer"])
        leniency_accuracy_scores.append(calculate_accuracy_some_leniency(predicted_value, sample["answer"]))
        exact_accuracy_scores.append(calculate_accuracy_exact(predicted_value, sample["answer"]))

    metric_results = evaluate_metrics(all_predictions, all_references)
    print(f"Average Accuracy with Leniency: {sum(leniency_accuracy_scores) / len(leniency_accuracy_scores):.2f}")
    print(f"Average Exact Accuracy: {sum(exact_accuracy_scores) / len(exact_accuracy_scores):.2f}")    
    print(f"ROUGE Score: {metric_results['rouge']:.2f}")
    print(f"BLEU Score: {metric_results['bleu']:.2f}")
    print(f"METEOR Score: {metric_results['meteor']:.2f}")
    print(f"BERTScore Precision: {metric_results['bertscore_precision']:.3f}")
    print(f"BERTScore Recall: {metric_results['bertscore_recall']:.3f}")
    print(f"BERTScore F1: {metric_results['bertscore_f1']:.3f}")
    print(f"GLEU Score: {metric_results['googlebleuscore']:.2f}")


# Test
def select_random_samples(dataset, num_samples, seed=None):
    if seed is not None:
        random.seed(seed)
    random_indices = random.sample(range(0, len(dataset)), num_samples)
    random_samples = dataset.select(random_indices)
    return random_samples


# Load your test dataset here
# test_set = ...

seed_value = 9
test_samples = select_random_samples(test_set, 10, seed=seed_value)

# Run inference and scoring for different numbers of examples
for num_examples in [0, 1, 2, 3, 4, 5]:
    print(f"\nRunning inference with {num_examples} example(s)")
    inference_and_scoring(test_set, model, tokenizer, num_examples=num_examples)

