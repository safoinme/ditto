#!/usr/bin/env python3
"""
Prepare CSV Data for Entity Matching

This script takes CSV data and prepares it for entity matching by:
1. Creating all possible pairs or specific pairs
2. Formatting them in the JSONL format expected by the matcher
3. Serializing entries in Ditto format

Usage examples:
    # Create all possible pairs from CSV
    python prepare_csv_for_matching.py data.csv --mode all_pairs --output input/all_pairs.jsonl

    # Create pairs from two separate CSV files
    python prepare_csv_for_matching.py data1.csv --second_csv data2.csv --output input/cross_pairs.jsonl

    # Create pairs from specific rows (by index)
    python prepare_csv_for_matching.py data.csv --mode specific_pairs --pairs "0,1;2,3;4,5" --output input/specific_pairs.jsonl

    # Deduplicate: find potential duplicates in a CSV
    python prepare_csv_for_matching.py data.csv --mode deduplicate --output input/dedupe_pairs.jsonl
"""

import pandas as pd
import json
import argparse
import os
import itertools
from pathlib import Path

class CSVMatchingPreparer:
    def __init__(self, csv_path, output_path="input/matching_pairs.jsonl"):
        self.csv_path = csv_path
        self.output_path = output_path
        self.df = pd.read_csv(csv_path)
        
        # Create output directory
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        
        print(f"Loaded CSV with {len(self.df)} records")
        print(f"Columns: {list(self.df.columns)}")
    
    def clean_text(self, text):
        """Clean and normalize text for Ditto format"""
        if pd.isna(text) or text == '':
            return ""
        return str(text).strip()
    
    def serialize_entry(self, row, row_index=None):
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
    
    def create_all_pairs(self):
        """Create all possible pairs from the CSV"""
        pairs = []
        total_pairs = len(self.df) * (len(self.df) - 1) // 2
        
        print(f"Creating all possible pairs... This will generate {total_pairs} pairs")
        
        if total_pairs > 10000:
            response = input(f"This will create {total_pairs} pairs. Continue? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return []
        
        pair_id = 0
        for i in range(len(self.df)):
            for j in range(i + 1, len(self.df)):
                entry1 = self.serialize_entry(self.df.iloc[i], i)
                entry2 = self.serialize_entry(self.df.iloc[j], j)
                
                pair = {
                    "id": f"pair_{pair_id}",
                    "left": entry1,
                    "right": entry2,
                    "left_index": i,
                    "right_index": j
                }
                pairs.append(pair)
                pair_id += 1
                
                if pair_id % 1000 == 0:
                    print(f"Created {pair_id}/{total_pairs} pairs...")
        
        print(f"Created {len(pairs)} pairs total")
        return pairs
    
    def create_cross_pairs(self, second_csv_path):
        """Create pairs between two different CSV files"""
        df2 = pd.read_csv(second_csv_path)
        print(f"Loaded second CSV with {len(df2)} records")
        
        pairs = []
        total_pairs = len(self.df) * len(df2)
        
        print(f"Creating cross pairs... This will generate {total_pairs} pairs")
        
        if total_pairs > 10000:
            response = input(f"This will create {total_pairs} pairs. Continue? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return []
        
        pair_id = 0
        for i in range(len(self.df)):
            for j in range(len(df2)):
                entry1 = self.serialize_entry(self.df.iloc[i], i)
                entry2 = self.serialize_entry(df2.iloc[j], j)
                
                pair = {
                    "id": f"cross_pair_{pair_id}",
                    "left": entry1,
                    "right": entry2,
                    "left_index": i,
                    "right_index": j,
                    "left_source": self.csv_path,
                    "right_source": second_csv_path
                }
                pairs.append(pair)
                pair_id += 1
                
                if pair_id % 1000 == 0:
                    print(f"Created {pair_id}/{total_pairs} pairs...")
        
        print(f"Created {len(pairs)} pairs total")
        return pairs
    
    def create_specific_pairs(self, pair_specs):
        """Create specific pairs based on row indices"""
        pairs = []
        
        # Parse pair specifications: "0,1;2,3;4,5"
        pair_list = []
        for spec in pair_specs.split(';'):
            try:
                idx1, idx2 = map(int, spec.split(','))
                if 0 <= idx1 < len(self.df) and 0 <= idx2 < len(self.df):
                    pair_list.append((idx1, idx2))
                else:
                    print(f"Warning: Invalid indices ({idx1}, {idx2}) - skipping")
            except ValueError:
                print(f"Warning: Invalid pair specification '{spec}' - skipping")
        
        print(f"Creating {len(pair_list)} specific pairs")
        
        for pair_id, (i, j) in enumerate(pair_list):
            entry1 = self.serialize_entry(self.df.iloc[i], i)
            entry2 = self.serialize_entry(self.df.iloc[j], j)
            
            pair = {
                "id": f"specific_pair_{pair_id}",
                "left": entry1,
                "right": entry2,
                "left_index": i,
                "right_index": j
            }
            pairs.append(pair)
        
        print(f"Created {len(pairs)} specific pairs")
        return pairs
    
    def create_deduplication_pairs(self, similarity_threshold=0.7):
        """Create pairs for deduplication (likely duplicates)"""
        pairs = []
        
        print("Creating deduplication pairs based on name similarity...")
        
        pair_id = 0
        for i in range(len(self.df)):
            for j in range(i + 1, len(self.df)):
                # Check if potentially similar
                if self.are_potentially_similar(self.df.iloc[i], self.df.iloc[j]):
                    entry1 = self.serialize_entry(self.df.iloc[i], i)
                    entry2 = self.serialize_entry(self.df.iloc[j], j)
                    
                    pair = {
                        "id": f"dedupe_pair_{pair_id}",
                        "left": entry1,
                        "right": entry2,
                        "left_index": i,
                        "right_index": j
                    }
                    pairs.append(pair)
                    pair_id += 1
        
        print(f"Created {len(pairs)} potential duplicate pairs")
        return pairs
    
    def are_potentially_similar(self, row1, row2):
        """Check if two rows are potentially similar (for deduplication)"""
        # Same IFU ID
        if (self.clean_text(row1['ifu_clean']) == self.clean_text(row2['ifu_clean']) and 
            self.clean_text(row1['ifu_clean']) != ""):
            return True
        
        # Similar last names
        lastname1 = self.clean_text(row1['nom_clean']).upper()
        lastname2 = self.clean_text(row2['nom_clean']).upper()
        
        if lastname1 and lastname2:
            # Exact match
            if lastname1 == lastname2:
                return True
            
            # Similar names (Levenshtein-like check)
            if len(lastname1) > 3 and len(lastname2) > 3:
                # Check if one is substring of another
                if lastname1 in lastname2 or lastname2 in lastname1:
                    return True
                
                # Check for common prefixes/suffixes
                if lastname1[:4] == lastname2[:4] or lastname1[-4:] == lastname2[-4:]:
                    return True
        
        return False
    
    def save_pairs(self, pairs):
        """Save pairs to JSONL file"""
        if not pairs:
            print("No pairs to save")
            return
        
        print(f"Saving {len(pairs)} pairs to {self.output_path}")
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                f.write(json.dumps(pair) + '\n')
        
        print(f"Saved successfully!")
        print(f"File: {self.output_path}")
        print(f"Pairs: {len(pairs)}")
        
        # Show sample
        if pairs:
            print("\nSample pair:")
            print(json.dumps(pairs[0], indent=2))
    
    def prepare_for_matching(self, mode="all_pairs", **kwargs):
        """Main method to prepare data for matching"""
        if mode == "all_pairs":
            pairs = self.create_all_pairs()
        elif mode == "cross_pairs":
            second_csv = kwargs.get('second_csv')
            if not second_csv:
                raise ValueError("second_csv path required for cross_pairs mode")
            pairs = self.create_cross_pairs(second_csv)
        elif mode == "specific_pairs":
            pair_specs = kwargs.get('pairs')
            if not pair_specs:
                raise ValueError("pairs specification required for specific_pairs mode")
            pairs = self.create_specific_pairs(pair_specs)
        elif mode == "deduplicate":
            pairs = self.create_deduplication_pairs()
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        self.save_pairs(pairs)
        return self.output_path

def main():
    parser = argparse.ArgumentParser(description='Prepare CSV data for entity matching')
    parser.add_argument('csv_file', help='Path to input CSV file')
    parser.add_argument('--mode', choices=['all_pairs', 'cross_pairs', 'specific_pairs', 'deduplicate'], 
                       default='deduplicate', help='Pairing mode (default: deduplicate)')
    parser.add_argument('--output', '-o', default='input/matching_pairs.jsonl', 
                       help='Output JSONL file path')
    parser.add_argument('--second_csv', help='Second CSV file for cross_pairs mode')
    parser.add_argument('--pairs', help='Pair specifications for specific_pairs mode (e.g., "0,1;2,3;4,5")')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file {args.csv_file} not found")
        return
    
    if args.mode == 'cross_pairs' and not args.second_csv:
        print("Error: --second_csv is required for cross_pairs mode")
        return
    
    if args.mode == 'specific_pairs' and not args.pairs:
        print("Error: --pairs is required for specific_pairs mode")
        return
    
    preparer = CSVMatchingPreparer(args.csv_file, args.output)
    
    kwargs = {}
    if args.second_csv:
        kwargs['second_csv'] = args.second_csv
    if args.pairs:
        kwargs['pairs'] = args.pairs
    
    output_file = preparer.prepare_for_matching(args.mode, **kwargs)
    
    print(f"\n{'='*60}")
    print("PREPARATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Output file: {output_file}")
    print("\nNext steps:")
    print("1. Run the matcher:")
    print(f"   python matcher.py --task person_records --input_path {output_file} --output_path output/predictions.jsonl --lm distilbert --max_len 128 --use_gpu --fp16 --checkpoint_path checkpoints/ --dk general --summarize")
    print("\n2. Or use the pipeline script:")
    print(f"   python run_person_matching.py --match-only --input {output_file}")

if __name__ == "__main__":
    main()