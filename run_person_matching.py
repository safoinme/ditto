#!/usr/bin/env python3
"""
Complete Person Record Matching Pipeline

This script runs the complete pipeline for person record entity matching:
1. Data augmentation
2. Convert to Ditto format  
3. Train the matching model
4. Run matching/prediction

Usage examples:
    # Run complete pipeline
    python run_person_matching.py --csv data.csv --complete

    # Just prepare data
    python run_person_matching.py --csv data.csv --prepare-only

    # Just train (if data already prepared)
    python run_person_matching.py --train-only

    # Just run matching
    python run_person_matching.py --match-only --input input/test_pairs.jsonl
"""

import argparse
import os
import subprocess
import sys
import json
from pathlib import Path

class PersonMatchingPipeline:
    def __init__(self, csv_path=None, output_dir="data/person_records"):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.augmented_csv = None
        self.model_dir = "checkpoints/person_records"
        
        # Create necessary directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path("input").mkdir(parents=True, exist_ok=True)
        Path("output").mkdir(parents=True, exist_ok=True)
    
    def run_command(self, cmd, description):
        """Run a shell command and handle errors"""
        print(f"\n{'='*60}")
        print(f"STEP: {description}")
        print(f"{'='*60}")
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stdout:
                print("Output:")
                print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            if e.stdout:
                print("stdout:", e.stdout)
            if e.stderr:
                print("stderr:", e.stderr)
            return False
    
    def augment_data(self):
        """Step 1: Augment the CSV data"""
        if not self.csv_path or not os.path.exists(self.csv_path):
            print(f"Error: CSV file {self.csv_path} not found")
            return False
        
        self.augmented_csv = self.csv_path.replace('.csv', '_augmented.csv')
        
        cmd = [
            sys.executable, "augment_csv_data.py",
            self.csv_path,
            "--output", self.augmented_csv,
            "--num_augmentations", "10"
        ]
        
        return self.run_command(cmd, "Data Augmentation")
    
    def convert_to_ditto(self):
        """Step 2: Convert augmented CSV to Ditto format"""
        csv_file = self.augmented_csv or self.csv_path
        
        cmd = [
            sys.executable, "csv_to_ditto.py",
            csv_file,
            "--output_dir", self.output_dir,
            "--match_ratio", "0.3"
        ]
        
        return self.run_command(cmd, "Convert to Ditto Format")
    
    def train_model(self):
        """Step 3: Train the Ditto model"""
        cmd = [
            sys.executable, "train_ditto.py",
            "--task", "person_records",
            "--batch_size", "16",
            "--max_len", "128",
            "--lr", "3e-5",
            "--n_epochs", "20",
            "--lm", "distilbert",
            "--fp16",
            "--da", "del",
            "--dk", "general",
            "--summarize",
            "--save_model",
            "--logdir", "checkpoints"
        ]
        
        return self.run_command(cmd, "Train Ditto Model")
    
    def create_sample_input(self):
        """Create a sample input file for matching"""
        sample_input_path = "input/sample_pairs.jsonl"
        
        # Read some data from the test set to create sample input
        test_file = os.path.join(self.output_dir, "test.txt")
        if not os.path.exists(test_file):
            print(f"Warning: Test file {test_file} not found, cannot create sample input")
            return False
        
        with open(test_file, 'r') as f:
            lines = f.readlines()[:10]  # Take first 10 pairs
        
        with open(sample_input_path, 'w') as f:
            for i, line in enumerate(lines):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    entry = {
                        "id": f"pair_{i}",
                        "left": parts[0],
                        "right": parts[1]
                    }
                    f.write(json.dumps(entry) + '\n')
        
        print(f"Created sample input file: {sample_input_path}")
        return True
    
    def run_matching(self, input_path="input/sample_pairs.jsonl", output_path="output/predictions.jsonl"):
        """Step 4: Run entity matching"""
        if not os.path.exists(input_path):
            print(f"Creating sample input file...")
            if not self.create_sample_input():
                return False
            input_path = "input/sample_pairs.jsonl"
        
        cmd = [
            sys.executable, "matcher.py",
            "--task", "person_records",
            "--input_path", input_path,
            "--output_path", output_path,
            "--lm", "distilbert",
            "--max_len", "128",
            "--use_gpu",
            "--fp16",
            "--checkpoint_path", "checkpoints/",
            "--dk", "general",
            "--summarize"
        ]
        
        return self.run_command(cmd, "Run Entity Matching")
    
    def run_complete_pipeline(self):
        """Run the complete pipeline"""
        print("Starting complete person record matching pipeline...")
        
        # Step 1: Augment data
        if not self.augment_data():
            print("Failed at data augmentation step")
            return False
        
        # Step 2: Convert to Ditto format
        if not self.convert_to_ditto():
            print("Failed at conversion step")
            return False
        
        # Step 3: Train model
        if not self.train_model():
            print("Failed at training step")
            return False
        
        # Step 4: Run matching
        if not self.run_matching():
            print("Failed at matching step")
            return False
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
        print(f"Model saved in: {self.model_dir}")
        print(f"Data files in: {self.output_dir}")
        print("Sample predictions in: output/predictions.jsonl")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Person Record Entity Matching Pipeline')
    parser.add_argument('--csv', help='Path to input CSV file')
    parser.add_argument('--complete', action='store_true', help='Run complete pipeline')
    parser.add_argument('--prepare-only', action='store_true', help='Only run data preparation steps')
    parser.add_argument('--train-only', action='store_true', help='Only run training')
    parser.add_argument('--match-only', action='store_true', help='Only run matching')
    parser.add_argument('--input', help='Input file for matching (JSONL format)')
    parser.add_argument('--output', default='output/predictions.jsonl', help='Output file for predictions')
    parser.add_argument('--output_dir', default='data/person_records', help='Directory for Ditto format files')
    
    args = parser.parse_args()
    
    if not any([args.complete, args.prepare_only, args.train_only, args.match_only]):
        parser.print_help()
        return
    
    pipeline = PersonMatchingPipeline(args.csv, args.output_dir)
    
    if args.complete:
        if not args.csv:
            print("Error: --csv is required for complete pipeline")
            return
        pipeline.run_complete_pipeline()
    
    elif args.prepare_only:
        if not args.csv:
            print("Error: --csv is required for data preparation")
            return
        pipeline.augment_data()
        pipeline.convert_to_ditto()
    
    elif args.train_only:
        pipeline.train_model()
    
    elif args.match_only:
        input_path = args.input or "input/sample_pairs.jsonl"
        pipeline.run_matching(input_path, args.output)

if __name__ == "__main__":
    main() 