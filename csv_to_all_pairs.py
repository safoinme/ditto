#!/usr/bin/env python3
"""
CSV to All Pairs Converter

This script takes a CSV file and creates all possible pairs between rows,
outputting them in JSONL format compatible with the matcher.

Usage:
    python csv_to_all_pairs.py input.csv --output pairs.jsonl
    python csv_to_all_pairs.py input.csv --output pairs.jsonl --limit 1000
"""

import pandas as pd
import json
import argparse
import os
from pathlib import Path
from tqdm import tqdm

def clean_text(text):
    """Clean and normalize text"""
    if pd.isna(text) or text == '':
        return ""
    return str(text).strip()

def row_to_dict(row, column_mapping=None):
    """Convert a pandas row to a clean dictionary"""
    result = {}
    
    # Default column mapping for person records (can be customized)
    if column_mapping is None:
        column_mapping = {
            'ifu_clean': 'ifu_id',
            'num_cin_clean': 'cin_number', 
            'num_ce_clean': 'ce_number',
            'num_ppr_clean': 'ppr_number',
            'num_cnss_clean': 'cnss_number',
            'nom_clean': 'lastname',
            'nom_prenom_rs_clean': 'fullname'
        }
    
    for col in row.index:
        value = clean_text(row[col])
        if value:  # Only include non-empty values
            clean_col_name = column_mapping.get(col, col)
            result[clean_col_name] = value
    
    return result

def csv_to_all_pairs(csv_path, output_path, limit=None, column_mapping=None, include_index=False):
    """
    Convert CSV to all possible pairs in JSONL format
    
    Args:
        csv_path (str): Path to input CSV file
        output_path (str): Path to output JSONL file
        limit (int, optional): Maximum number of pairs to generate
        column_mapping (dict, optional): Mapping of CSV columns to clean names
        include_index (bool): Whether to include row indices in output
    """
    
    # Load CSV
    print(f"Loading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records")
    
    # Calculate total possible pairs
    total_pairs = len(df) * (len(df) - 1)
    print(f"Total possible pairs: {total_pairs:,}")
    
    if limit and total_pairs > limit:
        print(f"Limiting to {limit:,} pairs")
        total_pairs = limit
    
    # Warn if this will create a very large file
    if total_pairs > 100000:
        estimated_size_mb = total_pairs * 0.5 / 1000  # Rough estimate
        response = input(f"This will create {total_pairs:,} pairs (~{estimated_size_mb:.1f}MB). Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return None
    
    # Create output directory
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    
    # Generate all pairs
    print(f"Generating pairs...")
    pairs_created = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Use tqdm for progress bar
        pbar = tqdm(total=total_pairs, desc="Creating pairs")
        
        for i in range(len(df)):
            for j in range(len(df)):
                if i != j:  # Don't pair a row with itself
                    # Convert rows to dictionaries
                    entity1 = row_to_dict(df.iloc[i], column_mapping)
                    entity2 = row_to_dict(df.iloc[j], column_mapping)
                    
                    # Add indices if requested
                    if include_index:
                        entity1['_row_index'] = i
                        entity2['_row_index'] = j
                    
                    # Create pair in the format expected by matcher: [entity1, entity2]
                    pair = [entity1, entity2]
                    
                    # Write to file
                    f.write(json.dumps(pair) + '\n')
                    pairs_created += 1
                    pbar.update(1)
                    
                    # Check limit
                    if limit and pairs_created >= limit:
                        pbar.close()
                        break
            
            if limit and pairs_created >= limit:
                break
        
        if not (limit and pairs_created >= limit):
            pbar.close()
    
    print(f"\n✅ Conversion complete!")
    print(f"📁 Output file: {output_path}")
    print(f"📊 Pairs created: {pairs_created:,}")
    print(f"💾 File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    # Show sample
    if pairs_created > 0:
        print(f"\n📋 Sample pair:")
        with open(output_path, 'r') as f:
            sample = json.loads(f.readline())
            print(f"Entity 1: {sample[0]}")
            print(f"Entity 2: {sample[1]}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Convert CSV to all possible pairs in JSONL format for matching')
    parser.add_argument('csv_file', help='Path to input CSV file')
    parser.add_argument('--output', '-o', default='output/all_pairs.jsonl', 
                       help='Output JSONL file path (default: output/all_pairs.jsonl)')
    parser.add_argument('--limit', '-l', type=int, 
                       help='Maximum number of pairs to generate (useful for large files)')
    parser.add_argument('--include-index', action='store_true',
                       help='Include row indices in the output pairs')
    parser.add_argument('--custom-mapping', type=str,
                       help='JSON file with custom column name mappings')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"❌ Error: File {args.csv_file} not found")
        return
    
    # Load custom column mapping if provided
    column_mapping = None
    if args.custom_mapping:
        if os.path.exists(args.custom_mapping):
            with open(args.custom_mapping, 'r') as f:
                column_mapping = json.load(f)
            print(f"Using custom column mapping from {args.custom_mapping}")
        else:
            print(f"⚠️  Warning: Custom mapping file {args.custom_mapping} not found, using defaults")
    
    output_file = csv_to_all_pairs(
        args.csv_file, 
        args.output, 
        limit=args.limit,
        column_mapping=column_mapping,
        include_index=args.include_index
    )
    
    if output_file:
        print(f"\n🚀 Next step - Run the matcher:")
        print(f"python matcher.py --task person_records --input_path {output_file} --output_path output/results.jsonl --lm distilbert --max_len 128 --checkpoint_path checkpoints/ --dk general --summarize")
        
        print(f"\n📖 Or use the pipeline:")
        print(f"python run_person_matching.py --match-only --input {output_file}")

if __name__ == "__main__":
    main() 