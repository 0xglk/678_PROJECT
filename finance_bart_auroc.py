import argparse
import json
import time
import openai
# OpenAI API key setup
client = openai.api_key=''
from bart_score import BARTScorer
import torch
import torch.nn as nn
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration
from typing import List
import numpy as np
from sklearn.metrics import roc_auc_score

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Run BART Scorer with optional commands.")
parser.add_argument("-c", "--custom-text", type=str, help="Print custom text and exit.")

args = parser.parse_args()

if args.custom_text:
    # Initialize the BARTScorer
    scorer = BARTScorer(device='cpu') 
    # Initialize the OpenAI client (ensure you've set up your API key correctly)
    client = openai.ChatCompletion()
    # Function to get revision from ChatGPT
    # Function to get revision from ChatGPT
    def get_revision(text):
        try:
            response = client.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Revise the following text:"},
                    {"role": "user", "content": text}
                ]
            )
            revised_text = response.choices[0].message['content']
            return revised_text
        except Exception as e:
            print(f"Error in getting revision from ChatGPT: {e}")
            return text  # Return original text if error

    # Function to compute similarity using BARTScore
    def compute_similarity(original, revised):
        try:
            scores = scorer.score([original], [revised])
            return scores[0]  # Assuming score method returns a list of scores
        except Exception as e:
            print(f"Error in computing similarity: {e}")
            return 0  # Return a neutral score in case of error

    # Main function to evaluate a text
    def evaluate_text(text):
        print(f"Original Text: {text}")
        revised_text = get_revision(text)
        print(f"Revised Text: {revised_text}")
        similarity_score = compute_similarity(text, revised_text)
        print(f"Similarity Score: {similarity_score}")

        # Define a threshold for determining LLM-generated text
        threshold = 0.5  # This value should be empirically determined based on validation data
        is_llm_generated = similarity_score >= threshold

        return is_llm_generated

    # Example usage
    original_text = args.custom_text
    is_generated = evaluate_text(original_text)
    print(f"Is the text LLM-generated? {'Yes' if is_generated else 'No'}")

    exit()

# Initializing BARTScorer
print("Initializing BARTScorer...")
bartscorer = BARTScorer(device='cpu', checkpoint="facebook/bart-large-cnn")
start_time = time.time()

# Open the files for reading
print("Reading revised human responses...")
with open("./revised_human_finance.txt", "r") as f:
    revised_human_list = [json.loads(line)[str(i)] for i, line in enumerate(f.readlines())]

print("Reading revised ChatGPT responses...")
with open("./revised_chatgpt_finance.txt", "r") as f2:
    revised_chatgpt_list = [json.loads(line)[str(i)] for i, line in enumerate(f2.readlines())]

chatgpt_list = []
human_list = []

# Load data from JSONL file
print("Loading human and ChatGPT responses from JSONL file...")
with open('./finance.jsonl', encoding="utf-8") as f:
    for row in f:
        data = json.loads(row)
        human_answers = data["human_answers"][0]
        chatgpt_answers = data["chatgpt_answers"][0]
        human_list.append(human_answers)
        chatgpt_list.append(chatgpt_answers)

chatcands = []
chatrefs = []
humancands = []
humanrefs = []

# Constructing input lists for BART scorer
print("Constructing input lists for BART scorer...")
for i in range(len(chatgpt_list)):
    text1 = revised_chatgpt_list[i]
    text2 = chatgpt_list[i]
    chatcands.append(text1)
    chatrefs.append(text2)
    
    text3 = revised_human_list[i]
    text4 = human_list[i]
    humancands.append(text3)
    humanrefs.append(text4)
    
    # Print current line being processed
    print("Processing line", i + 1, "of", len(chatgpt_list))

    elapsed_time = time.time() - start_time
    avg_time_per_item = elapsed_time / (i + 1)
    items_left = len(chatgpt_list) - (i + 1)
    time_left = avg_time_per_item * items_left
    print("Estimated time left:", round(time_left, 2), "seconds")

# Calculating scores using BART scorer
print("Calculating scores using BART scorer...")
start_bart_score_time = time.time()  # Record start time for BART score calculation
chatgpt_score = bartscorer.score(chatcands, chatrefs)
human_score = bartscorer.score(humancands, humanrefs)
end_bart_score_time = time.time()  # Record end time for BART score calculation

# Print processing time for BART score calculation
bart_score_processing_time = end_bart_score_time - start_bart_score_time
print("Processing time for calculating scores using BART scorer:", round(bart_score_processing_time, 2), "seconds")

# Print scores
print("Printing scores...")
print("ChatGPT scores:", chatgpt_score)
print("Human scores:", human_score)

# Constructing true labels and scores for ROC AUC calculation
print("Constructing true labels and scores for ROC AUC calculation...")
y_true = [1] * len(chatgpt_score) + [0] * len(human_score)
y_score = chatgpt_score + human_score

# Calculating ROC AUC score
print("Calculating ROC AUC score...")
auroc_score = roc_auc_score(y_true, y_score)

# Printing AUROC score and total process time
print("Printing AUROC score and total process time...")
print("The AUROC score is:", auroc_score)
print("Total process time:", round(time.time() - start_time, 2), "seconds")
