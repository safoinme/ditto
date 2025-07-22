#!/usr/bin/env python3
"""
Create training dataset from data_modified.csv with data augmentation.
Generates positive pairs (original + augmented) and negative pairs (non-matching).
Outputs in DITTO format with COL/VAL structure.
"""

import pandas as pd
import random
import re
import string
from typing import List, Dict, Tuple
import os

class DataAugmenter:
    """Data augmentation for entity matching dataset creation."""
    
    def __init__(self, seed=42):
        random.seed(seed)
        
        # Common abbreviations for names and addresses
        self.abbreviations = {
            'MOHAMMED': ['MED', 'MOHAMED', 'MOHD'],
            'ABDERRAHIM': ['ABDERRHIM', 'ABD'],
            'ABDELKADER': ['ABDELKDER', 'ABD'],
            'BOULEVARD': ['BD', 'BLVD'],
            'AVENUE': ['AV', 'AVE'],
            'RUE': ['R', 'ST'],
            'RESIDENCE': ['RES', 'RESID'],
            'IMMEUBLE': ['IMM', 'BLDG'],
            'APPARTEMENT': ['APPT', 'APT'],
            'NUMERO': ['N', 'NUM', 'NO'],
            'ETAGE': ['ETG', 'FLOOR'],
            'SOCIETE': ['SOC', 'STE'],
            'PRODUCTION': ['PROD'],
            'CONCEPT': ['CONC'],
            'LAHCEN': ['HSEN'],
            'HICHAM': ['HISHAM'],
            'ADNANE': ['ADNAN'],
        }
    
    def add_typo(self, text: str) -> str:
        """Add a single character typo (add, delete, or substitute)."""
        if not text or len(text) < 2:
            return text
            
        text = str(text)
        typo_type = random.choice(['add', 'delete', 'substitute'])
        pos = random.randint(0, len(text) - 1)
        
        if typo_type == 'add':
            char = random.choice(string.ascii_uppercase + string.digits)
            return text[:pos] + char + text[pos:]
        elif typo_type == 'delete' and len(text) > 1:
            return text[:pos] + text[pos+1:]
        elif typo_type == 'substitute':
            char = random.choice(string.ascii_uppercase + string.digits)
            return text[:pos] + char + text[pos+1:]
        
        return text
    
    def remove_word(self, text: str) -> str:
        """Remove a random word from text."""
        if not text:
            return text
            
        words = str(text).split()
        if len(words) <= 1:
            return text
            
        word_to_remove = random.randint(0, len(words) - 1)
        return ' '.join(words[:word_to_remove] + words[word_to_remove+1:])
    
    def abbreviate_text(self, text: str) -> str:
        """Apply abbreviations to text."""
        if not text:
            return text
            
        text = str(text).upper()
        for full_word, abbrevs in self.abbreviations.items():
            if full_word in text:
                abbrev = random.choice(abbrevs)
                text = text.replace(full_word, abbrev)
                break
        return text
    
    def make_null(self, text: str) -> str:
        """Make field null/empty."""
        return ""
    
    def augment_record(self, record: Dict) -> Dict:
        """Augment a single record by modifying 1-2 fields."""
        augmented = record.copy()
        
        # Select fields to augment (excluding ID fields)
        augmentable_fields = [col for col in record.keys() 
                            if col not in ['ifu'] and pd.notna(record[col]) and str(record[col]).strip()]
        
        if not augmentable_fields:
            return augmented
        
        # Randomly select 1-2 fields to augment
        num_fields_to_augment = random.choice([1, 2])
        fields_to_augment = random.sample(augmentable_fields, 
                                        min(num_fields_to_augment, len(augmentable_fields)))
        
        for field in fields_to_augment:
            original_value = str(record[field])
            
            # Choose augmentation strategy based on field type and content
            if field in ['nom', 'prenoms', 'raison_sociale', 'nom_prenom_rs']:
                # For names and company names: typo, abbreviation, or word removal
                strategy = random.choice(['typo', 'abbreviate', 'remove_word'])
            elif field in ['adresse']:
                # For addresses: typo, abbreviation, or word removal
                strategy = random.choice(['typo', 'abbreviate', 'remove_word'])
            elif field in ['email_adherent']:
                # For emails: only typos
                strategy = 'typo'
            else:
                # For other fields: typo or make null
                strategy = random.choice(['typo', 'null'])
            
            if strategy == 'typo':
                augmented[field] = self.add_typo(original_value)
            elif strategy == 'abbreviate':
                augmented[field] = self.abbreviate_text(original_value)
            elif strategy == 'remove_word':
                augmented[field] = self.remove_word(original_value)
            elif strategy == 'null':
                augmented[field] = self.make_null(original_value)
        
        return augmented

class DittoDatasetCreator:
    """Create DITTO format dataset from CSV data."""
    
    def __init__(self, csv_path: str, output_dir: str = "./data"):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.augmenter = DataAugmenter()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} records from {csv_path}")
    
    def record_to_ditto_format(self, record: Dict) -> str:
        """Convert a record to DITTO COL/VAL format."""
        parts = []
        for col, val in record.items():
            if pd.notna(val) and str(val).strip():
                parts.append(f"COL {col} VAL {val}")
        return " ".join(parts)
    
    def create_positive_pairs(self) -> List[Tuple[str, str, int]]:
        """Create positive pairs: original + augmented."""
        positive_pairs = []
        
        for _, record in self.df.iterrows():
            original = record.to_dict()
            augmented = self.augmenter.augment_record(original)
            
            original_str = self.record_to_ditto_format(original)
            augmented_str = self.record_to_ditto_format(augmented)
            
            positive_pairs.append((original_str, augmented_str, 1))
        
        print(f"Created {len(positive_pairs)} positive pairs")
        return positive_pairs
    
    def create_negative_pairs(self, num_negative: int) -> List[Tuple[str, str, int]]:
        """Create negative pairs: random non-matching records."""
        negative_pairs = []
        records = [self.record_to_ditto_format(row.to_dict()) for _, row in self.df.iterrows()]
        
        for _ in range(num_negative):
            # Pick two different random records
            idx1, idx2 = random.sample(range(len(records)), 2)
            negative_pairs.append((records[idx1], records[idx2], 0))
        
        print(f"Created {len(negative_pairs)} negative pairs")
        return negative_pairs
    
    def create_dataset(self, negative_ratio: float = 1.0):
        """Create complete dataset with train/test/val splits."""
        
        # Create positive pairs
        positive_pairs = self.create_positive_pairs()
        
        # Create negative pairs (same number as positive by default)
        num_negative = int(len(positive_pairs) * negative_ratio)
        negative_pairs = self.create_negative_pairs(num_negative)
        
        # Combine and shuffle
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)
        
        print(f"Total dataset size: {len(all_pairs)} pairs")
        print(f"Positive pairs: {len(positive_pairs)} ({len(positive_pairs)/len(all_pairs)*100:.1f}%)")
        print(f"Negative pairs: {len(negative_pairs)} ({len(negative_pairs)/len(all_pairs)*100:.1f}%)")
        
        # Split into train/test/val (0.8/0.1/0.1)
        total_size = len(all_pairs)
        train_size = int(total_size * 0.8)
        test_size = int(total_size * 0.1)
        val_size = total_size - train_size - test_size
        
        train_pairs = all_pairs[:train_size]
        test_pairs = all_pairs[train_size:train_size + test_size]
        val_pairs = all_pairs[train_size + test_size:]
        
        print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}, Val: {len(val_pairs)}")
        
        # Write to files
        self.write_pairs_to_file(train_pairs, os.path.join(self.output_dir, "train.txt"))
        self.write_pairs_to_file(test_pairs, os.path.join(self.output_dir, "test.txt"))
        self.write_pairs_to_file(val_pairs, os.path.join(self.output_dir, "valid.txt"))
        
        print(f"Dataset files created in {self.output_dir}/")
    
    def write_pairs_to_file(self, pairs: List[Tuple[str, str, int]], filepath: str):
        """Write pairs to file in DITTO format."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for left, right, label in pairs:
                f.write(f"{left}\t{right}\t{label}\n")
        print(f"Written {len(pairs)} pairs to {filepath}")

def main():
    # Configuration
    csv_file = "data_modified.csv"
    output_directory = "./data/person_records"
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return
    
    # Create dataset
    creator = DittoDatasetCreator(csv_file, output_directory)
    creator.create_dataset(negative_ratio=1.0)  # 1:1 ratio of positive to negative pairs
    
    print("\nDataset creation completed!")
    print("Files created:")
    for filename in ["train.txt", "test.txt", "valid.txt"]:
        filepath = os.path.join(output_directory, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                line_count = sum(1 for line in f)
            print(f"  {filepath}: {line_count} pairs")

if __name__ == "__main__":
    main()