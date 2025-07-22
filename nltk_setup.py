#!/usr/bin/env python3
"""
NLTK setup helper for offline environments.
"""
import os
import nltk

def setup_nltk_data():
    """Setup NLTK data path for offline use."""
    # Check if NLTK_DATA environment variable is set
    nltk_data_path = os.getenv('NLTK_DATA', './nltk_data')
    
    # Add the path to NLTK data search paths if it exists
    if os.path.exists(nltk_data_path):
        if nltk_data_path not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_path)
            print(f"Added NLTK data path: {nltk_data_path}")
    
    # Verify stopwords are available
    try:
        from nltk.corpus import stopwords
        stopwords.words('english')[:5]  # Test access
        print("NLTK stopwords loaded successfully")
        return True
    except Exception as e:
        print(f"Warning: Could not load NLTK stopwords: {e}")
        return False

if __name__ == "__main__":
    setup_nltk_data()