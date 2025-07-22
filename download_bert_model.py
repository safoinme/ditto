#!/usr/bin/env python3
"""
Download BERT model files and NLTK data for offline use in airgapped environment.
"""
import os
import nltk
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

def download_nltk_data():
    # Set NLTK data path to local directory
    nltk_data_dir = "./nltk_data"
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Add the local directory to NLTK data path
    nltk.data.path.append(nltk_data_dir)
    
    print(f"Downloading NLTK data to {nltk_data_dir}...")
    
    # Download stopwords
    nltk.download('stopwords', download_dir=nltk_data_dir)
    
    # Download other commonly used NLTK datasets
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('wordnet', download_dir=nltk_data_dir)
    nltk.download('omw-1.4', download_dir=nltk_data_dir)
    
    print("NLTK data downloaded successfully")
    print(f"NLTK data saved to: {nltk_data_dir}")
    
    # Print downloaded contents
    for root, dirs, files in os.walk(nltk_data_dir):
        level = root.replace(nltk_data_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Limit to first 5 files per directory
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")

def download_spacy_model():
    """Download spaCy model for domain knowledge injection."""
    import subprocess
    import sys
    
    print("Downloading spaCy en_core_web_lg model...")
    try:
        # Download the model
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_lg"])
        print("✓ spaCy en_core_web_lg model downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to download spaCy model: {e}")
        return False
    except Exception as e:
        print(f"✗ Error downloading spaCy model: {e}")
        return False

if __name__ == "__main__":
    download_bert_model()
    download_nltk_data()
    download_spacy_model()