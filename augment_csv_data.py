#!/usr/bin/env python3
"""
Data Augmentation Script for Person Records CSV

This script takes a CSV file with person records and creates augmented data
by randomly modifying 1-2 fields per row with character-level changes.
It generates 10 new examples for each original line.
"""

import pandas as pd
import random
import string
import argparse
import os
from pathlib import Path

class CSVDataAugmenter:
    def __init__(self, input_csv_path, output_csv_path=None):
        self.input_csv_path = input_csv_path
        self.output_csv_path = output_csv_path or input_csv_path.replace('.csv', '_augmented.csv')
        self.df = pd.read_csv(input_csv_path)
        
    def add_random_char(self, text):
        """Add a random letter to the text at a random position"""
        if pd.isna(text) or text == '':
            return text
        text = str(text)
        pos = random.randint(0, len(text))
        char = random.choice(string.ascii_uppercase)
        return text[:pos] + char + text[pos:]
    
    def remove_random_chars(self, text, num_chars=1):
        """Remove 1-2 random characters from the text"""
        if pd.isna(text) or text == '' or len(str(text)) <= num_chars:
            return text
        text = str(text)
        for _ in range(min(num_chars, len(text))):
            if len(text) > 1:
                pos = random.randint(0, len(text) - 1)
                text = text[:pos] + text[pos + 1:]
        return text
    
    def change_random_chars(self, text, num_chars=1):
        """Change 1-2 random characters in the text"""
        if pd.isna(text) or text == '':
            return text
        text = str(text)
        text_list = list(text)
        positions = random.sample(range(len(text_list)), min(num_chars, len(text_list)))
        
        for pos in positions:
            if text_list[pos].isalpha():
                text_list[pos] = random.choice(string.ascii_uppercase)
            elif text_list[pos].isdigit():
                text_list[pos] = random.choice(string.digits)
        
        return ''.join(text_list)
    
    def augment_field(self, text):
        """Apply one of the augmentation operations to a field"""
        operations = [
            lambda x: self.add_random_char(x),
            lambda x: self.remove_random_chars(x, 1),
            lambda x: self.remove_random_chars(x, 2),
            lambda x: self.change_random_chars(x, 1),
            lambda x: self.change_random_chars(x, 2)
        ]
        
        operation = random.choice(operations)
        return operation(text)
    
    def augment_row(self, row):
        """Augment a single row by modifying 1-2 fields"""
        new_row = row.copy()
        
        # Get field names excluding empty ones
        non_empty_fields = [col for col in row.index 
                          if not pd.isna(row[col]) and str(row[col]).strip() != '']
        
        if len(non_empty_fields) == 0:
            return new_row
        
        # Randomly choose 1 or 2 fields to modify
        num_fields_to_modify = random.choice([1, 2])
        num_fields_to_modify = min(num_fields_to_modify, len(non_empty_fields))
        
        fields_to_modify = random.sample(non_empty_fields, num_fields_to_modify)
        
        for field in fields_to_modify:
            new_row[field] = self.augment_field(row[field])
        
        return new_row
    
    def generate_augmented_data(self, num_augmentations_per_row=10):
        """Generate augmented data for the entire dataset"""
        augmented_rows = []
        
        # Keep original data
        for _, row in self.df.iterrows():
            augmented_rows.append(row)
        
        # Generate augmented data
        print(f"Generating {num_augmentations_per_row} augmentations for each of {len(self.df)} rows...")
        
        for idx, row in self.df.iterrows():
            for aug_num in range(num_augmentations_per_row):
                augmented_row = self.augment_row(row)
                augmented_rows.append(augmented_row)
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(self.df)} rows")
        
        # Create new DataFrame
        augmented_df = pd.DataFrame(augmented_rows)
        return augmented_df
    
    def save_augmented_data(self, num_augmentations_per_row=10):
        """Generate and save augmented data"""
        augmented_df = self.generate_augmented_data(num_augmentations_per_row)
        
        print(f"Saving augmented data to {self.output_csv_path}")
        augmented_df.to_csv(self.output_csv_path, index=False)
        
        print(f"Original dataset size: {len(self.df)} rows")
        print(f"Augmented dataset size: {len(augmented_df)} rows")
        print(f"Total augmentations created: {len(augmented_df) - len(self.df)} rows")
        
        return self.output_csv_path

def main():
    parser = argparse.ArgumentParser(description='Augment CSV data with character-level modifications')
    parser.add_argument('input_csv', help='Path to input CSV file')
    parser.add_argument('--output', '-o', help='Path to output CSV file (default: input_augmented.csv)')
    parser.add_argument('--num_augmentations', '-n', type=int, default=10, 
                       help='Number of augmentations per original row (default: 10)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_csv):
        print(f"Error: Input file {args.input_csv} does not exist")
        return
    
    output_path = args.output or args.input_csv.replace('.csv', '_augmented.csv')
    
    augmenter = CSVDataAugmenter(args.input_csv, output_path)
    augmenter.save_augmented_data(args.num_augmentations)

if __name__ == "__main__":
    main() 