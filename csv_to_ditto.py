#!/usr/bin/env python3
"""
CSV to Ditto Format Converter

This script converts person records CSV data into Ditto entity matching format.
It creates positive (matching) and negative (non-matching) pairs and splits
the data into train/validation/test sets.
"""

import pandas as pd
import random
import argparse
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

class CSVToDittoConverter:
    def __init__(self, input_csv_path, output_dir="data/person_records"):
        self.input_csv_path = input_csv_path
        self.output_dir = output_dir
        self.df = pd.read_csv(input_csv_path)
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    def clean_text(self, text):
        """Clean and normalize text for Ditto format"""
        if pd.isna(text) or text == '':
            return ""
        return str(text).strip()
    
    def serialize_entry(self, row):
        """Convert a row to Ditto serialized format"""
        parts = []
        
        # Map column names to more readable format
        col_mapping = {
            'ifu_clean': 'ifu_id',
            'num_cin_clean': 'cin_number', 
            'num_ce_clean': 'ce_number',
            'num_ppr_clean': 'ppr_number',
            'num_cnss_clean': 'cnss_number',
            'nom_clean': 'lastname',
            'nom_prenom_rs_clean': 'fullname'
        }
        
        for col in self.df.columns:
            value = self.clean_text(row[col])
            if value:  # Only include non-empty values
                col_name = col_mapping.get(col, col)
                parts.append(f"COL {col_name} VAL {value}")
        
        return " ".join(parts)
    
    def are_likely_matches(self, row1, row2):
        """
        Determine if two rows are likely to be the same person.
        This creates positive matching pairs based on similarity.
        """
        # Check for exact matches in key identifying fields
        if (self.clean_text(row1['ifu_clean']) == self.clean_text(row2['ifu_clean']) and 
            self.clean_text(row1['ifu_clean']) != ""):
            return True
            
        # Check for similar names (same last name and similar full name)
        lastname1 = self.clean_text(row1['nom_clean']).upper()
        lastname2 = self.clean_text(row2['nom_clean']).upper()
        fullname1 = self.clean_text(row1['nom_prenom_rs_clean']).upper()
        fullname2 = self.clean_text(row2['nom_prenom_rs_clean']).upper()
        
        if lastname1 and lastname2 and lastname1 == lastname2:
            # Check if fullnames are similar (more than 70% overlap)
            if fullname1 and fullname2:
                # Simple similarity check - count common words
                words1 = set(fullname1.split())
                words2 = set(fullname2.split())
                if len(words1) > 0 and len(words2) > 0:
                    overlap = len(words1.intersection(words2))
                    similarity = overlap / max(len(words1), len(words2))
                    if similarity > 0.7:
                        return True
        
        return False
    
    def create_entity_pairs(self, match_ratio=0.3):
        """
        Create positive and negative entity matching pairs.
        
        Args:
            match_ratio: Ratio of positive to negative examples
        """
        pairs = []
        labels = []
        
        print(f"Creating entity matching pairs from {len(self.df)} records...")
        
        # Create positive pairs (matches)
        positive_pairs = 0
        for i in range(len(self.df)):
            for j in range(i + 1, len(self.df)):
                if self.are_likely_matches(self.df.iloc[i], self.df.iloc[j]):
                    entry1 = self.serialize_entry(self.df.iloc[i])
                    entry2 = self.serialize_entry(self.df.iloc[j])
                    pairs.append((entry1, entry2))
                    labels.append(1)
                    positive_pairs += 1
        
        print(f"Created {positive_pairs} positive pairs")
        
        # Create negative pairs (non-matches)
        target_negative_pairs = int(positive_pairs / match_ratio) - positive_pairs
        negative_pairs = 0
        
        while negative_pairs < target_negative_pairs:
            i = random.randint(0, len(self.df) - 1)
            j = random.randint(0, len(self.df) - 1)
            
            if i != j and not self.are_likely_matches(self.df.iloc[i], self.df.iloc[j]):
                entry1 = self.serialize_entry(self.df.iloc[i])
                entry2 = self.serialize_entry(self.df.iloc[j])
                pairs.append((entry1, entry2))
                labels.append(0)
                negative_pairs += 1
        
        print(f"Created {negative_pairs} negative pairs")
        print(f"Total pairs: {len(pairs)}")
        print(f"Positive ratio: {positive_pairs / len(pairs):.2%}")
        
        return pairs, labels
    
    def split_and_save_data(self, pairs, labels, train_ratio=0.7, val_ratio=0.15):
        """Split data into train/validation/test sets and save to files"""
        
        # First split: train vs (val + test)
        train_pairs, temp_pairs, train_labels, temp_labels = train_test_split(
            pairs, labels, test_size=(1 - train_ratio), stratify=labels, random_state=42
        )
        
        # Second split: val vs test
        val_test_split = val_ratio / (1 - train_ratio)
        val_pairs, test_pairs, val_labels, test_labels = train_test_split(
            temp_pairs, temp_labels, test_size=(1 - val_test_split), 
            stratify=temp_labels, random_state=42
        )
        
        # Save datasets
        datasets = [
            (train_pairs, train_labels, "train.txt"),
            (val_pairs, val_labels, "valid.txt"), 
            (test_pairs, test_labels, "test.txt")
        ]
        
        for data_pairs, data_labels, filename in datasets:
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                for (entry1, entry2), label in zip(data_pairs, data_labels):
                    f.write(f"{entry1}\t{entry2}\t{label}\n")
            
            pos_count = sum(data_labels)
            total_count = len(data_labels)
            print(f"Saved {filename}: {total_count} pairs ({pos_count} positive, {total_count - pos_count} negative)")
        
        return {
            'train': os.path.join(self.output_dir, "train.txt"),
            'valid': os.path.join(self.output_dir, "valid.txt"),
            'test': os.path.join(self.output_dir, "test.txt")
        }
    
    def convert(self, match_ratio=0.3):
        """Main conversion method"""
        print(f"Converting {self.input_csv_path} to Ditto format...")
        print(f"Dataset shape: {self.df.shape}")
        
        # Create entity matching pairs
        pairs, labels = self.create_entity_pairs(match_ratio)
        
        # Split and save data
        filepaths = self.split_and_save_data(pairs, labels)
        
        print(f"\nConversion complete! Files saved to {self.output_dir}/")
        print("Files created:")
        for split, path in filepaths.items():
            print(f"  {split}: {path}")
        
        return filepaths

def main():
    parser = argparse.ArgumentParser(description='Convert CSV data to Ditto entity matching format')
    parser.add_argument('input_csv', help='Path to input CSV file')
    parser.add_argument('--output_dir', '-o', default='data/person_records',
                       help='Output directory for Ditto format files (default: data/person_records)')
    parser.add_argument('--match_ratio', '-r', type=float, default=0.3,
                       help='Ratio of positive to negative examples (default: 0.3)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_csv):
        print(f"Error: Input file {args.input_csv} does not exist")
        return
    
    converter = CSVToDittoConverter(args.input_csv, args.output_dir)
    converter.convert(args.match_ratio)

if __name__ == "__main__":
    main() 