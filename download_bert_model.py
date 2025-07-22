#!/usr/bin/env python3
"""
Download BERT model files for offline use in airgapped environment.
"""
import os
from transformers import AutoTokenizer, AutoModel

def download_bert_model():
    model_name = "bert-base-uncased"
    local_dir = "./models/bert-base-uncased"
    
    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"Downloading {model_name} to {local_dir}...")
    
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(local_dir)
    
    # Download model
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(local_dir)
    
    print(f"Model downloaded successfully to {local_dir}")
    print("Files downloaded:")
    for file in os.listdir(local_dir):
        print(f"  - {file}")

if __name__ == "__main__":
    download_bert_model()