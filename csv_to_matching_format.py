#!/usr/bin/env python3
"""
Convert CSV to Matching Format

Simple script that takes your CSV data and converts it to the format 
your trained model expects for entity matching.

Usage:
    python csv_to_matching_format.py data.csv --output input/pairs_to_match.jsonl
"""

import pandas as pd
import json
import argparse
import os
from pathlib import Path

def clean_text(text):
    """Clean and normalize text"""
    if pd.isna(text) or text == '':
        return ""
    return str(text).strip()

def serialize_entry(row, df_columns):
    """Convert a CSV row to Ditto serialized format"""
    parts = []
    
    # Map your CSV columns to clean names
    col_mapping = {
        'ifu_clean': 'ifu_id',
        'num_cin_clean': 'cin_number', 
        'num_ce_clean': 'ce_number',
        'num_ppr_clean': 'ppr_number',
        'num_cnss_clean': 'cnss_number',
        'nom_clean': 'lastname',
        'nom_prenom_rs_clean': 'fullname'
    }
    
    for col in df_columns:
        value = clean_text(row[col])
        if value:  # Only include non-empty values
            col_name = col_mapping.get(col, col)
            parts.append(f"COL {col_name} VAL {value}")
    
    return " ".join(parts)

def csv_to_matching_format(csv_path, output_path, mode="dedupe"):
    """Convert CSV to matching format"""
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")
    
    # Create output directory
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    
    pairs = []
    
    if mode == "dedupe":
        # Create pairs for deduplication (find duplicates within same file)
        print("Creating deduplication pairs...")
        pair_id = 0
        
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                # Check if potentially similar (same last name or similar)
                lastname1 = clean_text(df.iloc[i]['nom_clean']).upper()
                lastname2 = clean_text(df.iloc[j]['nom_clean']).upper()
                
                # Only create pairs for potentially similar records
                if (lastname1 and lastname2 and 
                    (lastname1 == lastname2 or 
                     (len(lastname1) > 3 and len(lastname2) > 3 and 
                      (lastname1 in lastname2 or lastname2 in lastname1)))):
                    
                    entry1 = serialize_entry(df.iloc[i], df.columns)
                    entry2 = serialize_entry(df.iloc[j], df.columns)
                    
                    pair = {
                        "id": f"pair_{pair_id}",
                        "left": entry1,
                        "right": entry2
                    }
                    pairs.append(pair)
                    pair_id += 1
    
    elif mode == "all":
        # Create all possible pairs
        print("Creating all possible pairs...")
        total_pairs = len(df) * (len(df) - 1) // 2
        print(f"This will create {total_pairs} pairs")
        
        if total_pairs > 5000:
            response = input(f"This will create {total_pairs} pairs. Continue? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return None
        
        pair_id = 0
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                entry1 = serialize_entry(df.iloc[i], df.columns)
                entry2 = serialize_entry(df.iloc[j], df.columns)
                
                pair = {
                    "id": f"pair_{pair_id}",
                    "left": entry1,
                    "right": entry2
                }
                pairs.append(pair)
                pair_id += 1
                
                if pair_id % 1000 == 0:
                    print(f"Created {pair_id} pairs...")
    
    # Save to JSONL file
    print(f"Saving {len(pairs)} pairs to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')
    
    print(f"✅ Conversion complete!")
    print(f"📁 Output: {output_path}")
    print(f"📊 Pairs created: {len(pairs)}")
    
    if pairs:
        print(f"\n📋 Sample pair:")
        print(f"Left:  {pairs[0]['left']}")
        print(f"Right: {pairs[0]['right']}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Convert CSV to format for trained matching model')
    parser.add_argument('csv_file', help='Path to your CSV file')
    parser.add_argument('--output', '-o', default='input/pairs_to_match.jsonl', 
                       help='Output file path (default: input/pairs_to_match.jsonl)')
    parser.add_argument('--mode', choices=['dedupe', 'all'], default='dedupe',
                       help='dedupe: only similar pairs, all: all possible pairs (default: dedupe)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"❌ Error: File {args.csv_file} not found")
        return
    
    output_file = csv_to_matching_format(args.csv_file, args.output, args.mode)
    
    if output_file:
        print(f"\n🚀 Next step - Run the matcher:")
        print(f"python matcher.py --task person_records --input_path {output_file} --output_path output/results.jsonl --lm distilbert --max_len 128 --checkpoint_path checkpoints/ --dk general --summarize")
        
        print(f"\n📖 Or use the pipeline:")
        print(f"python run_person_matching.py --match-only --input {output_file}")

if __name__ == "__main__":
    main() 
    
    
    