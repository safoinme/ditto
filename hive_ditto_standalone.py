#!/usr/bin/env python3
"""
Standalone Hive-DITTO integration script that works exactly like the notebook.
This script can be used independently or as part of Kubeflow pipelines.
"""

import os
import json
import pandas as pd
import numpy as np
import subprocess
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HiveConnector:
    """Handle Hive database connections and operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection = None
    
    def connect(self):
        """Establish connection to Hive."""
        try:
            from pyhive import hive
            self.connection = hive.Connection(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                username=self.config.get('username'),
                auth=self.config.get('auth', 'NOSASL')
            )
            logger.info(f"Connected to Hive: {self.config['host']}:{self.config['port']}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Hive: {e}")
            return False
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame."""
        if not self.connection:
            raise RuntimeError("Not connected to Hive")
        
        try:
            df = pd.read_sql(query, self.connection)
            logger.info(f"Query executed successfully, returned {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    def close(self):
        """Close Hive connection."""
        if self.connection:
            self.connection.close()
            logger.info("Hive connection closed")


def detect_table_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect if table has _left/_right columns (production) or single columns (testing).
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with structure information
    """
    columns = list(df.columns)
    
    # Remove table prefixes first to analyze column structure
    clean_columns = []
    for col in columns:
        if '.' in col:
            clean_col = col.split('.', 1)[1]
        else:
            clean_col = col
        clean_columns.append(clean_col)
    
    # Check for _left and _right patterns
    left_columns = [col for col in clean_columns if col.endswith('_left')]
    right_columns = [col for col in clean_columns if col.endswith('_right')]
    
    # Extract base field names
    left_fields = {col[:-5] for col in left_columns}  # Remove '_left'
    right_fields = {col[:-6] for col in right_columns}  # Remove '_right'
    
    # Check if we have matching left/right pairs
    matching_fields = left_fields.intersection(right_fields)
    
    if matching_fields:
        structure_type = "production"
        message = f"🏭 Production table detected with {len(matching_fields)} matching field pairs"
    else:
        structure_type = "testing"
        message = f"🧪 Testing table detected with {len(clean_columns)} fields for self-matching"
    
    return {
        'type': structure_type,
        'columns': columns,
        'clean_columns': clean_columns,
        'left_columns': left_columns,
        'right_columns': right_columns,
        'matching_fields': list(matching_fields),
        'message': message
    }

def convert_production_format(df: pd.DataFrame, structure: Dict) -> List[Dict[str, Any]]:
    """Convert production table with _left/_right columns to DITTO format."""
    records = []
    
    for idx, row in df.iterrows():
        # Build left record
        left_parts = []
        right_parts = []
        
        for field in structure['matching_fields']:
            # Find the actual column names (with potential table prefixes)
            left_col = None
            right_col = None
            
            for col in df.columns:
                clean_col = col.split('.', 1)[1] if '.' in col else col
                if clean_col == f"{field}_left":
                    left_col = col
                elif clean_col == f"{field}_right":
                    right_col = col
            
            # Process left column
            if left_col and pd.notna(row[left_col]) and str(row[left_col]).strip():
                value = str(row[left_col]).strip()
                left_parts.append(f"COL {field} VAL {value}")
                if idx == 0:  # Show transformation for first record
                    print(f"🔧 Left: {left_col} -> {field}")
            
            # Process right column
            if right_col and pd.notna(row[right_col]) and str(row[right_col]).strip():
                value = str(row[right_col]).strip()
                right_parts.append(f"COL {field} VAL {value}")
                if idx == 0:  # Show transformation for first record
                    print(f"🔧 Right: {right_col} -> {field}")
        
        left_text = " ".join(left_parts)
        right_text = " ".join(right_parts)
        
        # Create JSONL record for matching
        record = {
            "left": left_text,
            "right": right_text,
            "id": idx
        }
        records.append(record)
    
    print(f"✅ Successfully converted {len(records)} production records with left/right pairs")
    return records

def convert_testing_format(df: pd.DataFrame, structure: Dict) -> List[Dict[str, Any]]:
    """Convert testing table for self-matching to DITTO format."""
    records = []
    
    for idx, row in df.iterrows():
        # Convert row to COL/VAL format
        col_val_parts = []
        
        for col in df.columns:
            if pd.notna(row[col]) and str(row[col]).strip():
                # Remove table name prefix if present
                if '.' in col:
                    # Split by first dot only, in case there are multiple dots
                    clean_col = col.split('.', 1)[1]
                    print(f"🔧 Column: {col} -> {clean_col}") if idx == 0 else None  # Show for first record only
                else:
                    clean_col = col
                
                value = str(row[col]).strip()
                col_val_parts.append(f"COL {clean_col} VAL {value}")
        
        record_text = " ".join(col_val_parts)
        
        # Create JSONL record for self-matching
        record = {
            "left": record_text,
            "right": record_text,  # Same record for self-matching
            "id": idx
        }
        records.append(record)
    
    print(f"✅ Successfully converted {len(records)} testing records for self-matching")
    
    # Show column transformation summary
    if any('.' in col for col in df.columns):
        print("\n📋 Column transformations applied:")
        sample_cols = [col for col in df.columns if '.' in col][:5]  # Show first 5
        for col in sample_cols:
            clean_col = col.split('.', 1)[1]
            print(f"  {col} → {clean_col}")
        if len([col for col in df.columns if '.' in col]) > 5:
            remaining = len([col for col in df.columns if '.' in col]) - 5
            print(f"  ... and {remaining} more columns")
    
    return records

def convert_to_ditto_format(df: pd.DataFrame, matching_mode: str = 'auto') -> List[Dict[str, Any]]:
    """
    Convert DataFrame to DITTO format, handling both production (_left/_right) and testing (self-match) scenarios.
    
    Args:
        df: DataFrame with columns that may have tablename.column format
        matching_mode: 'auto', 'production', or 'testing'
        
    Returns:
        List of DITTO records in JSONL format
    """
    records = []
    
    print(f"🔍 Original columns: {list(df.columns)}")
    
    # Detect table structure
    structure = detect_table_structure(df)
    
    # Override detection if mode is specified
    if matching_mode == 'production':
        structure['type'] = 'production'
        structure['message'] = f"🏭 Forced production mode with {len(structure['matching_fields'])} matching field pairs"
    elif matching_mode == 'testing':
        structure['type'] = 'testing'
        structure['message'] = f"🧪 Forced testing mode with {len(structure['clean_columns'])} fields for self-matching"
    
    print(structure['message'])
    
    # Show sample of table prefixes detected
    table_prefixes = set()
    for col in df.columns:
        if '.' in col:
            prefix = col.split('.')[0]
            table_prefixes.add(prefix)
    
    if table_prefixes:
        print(f"🏷️  Detected table prefixes: {list(table_prefixes)}")
    else:
        print("ℹ️  No table prefixes detected")
    
    if structure['type'] == 'production':
        if not structure['matching_fields']:
            raise ValueError("Production mode requires _left/_right column pairs, but none were found!")
        print(f"📊 Matching fields: {structure['matching_fields']}")
        return convert_production_format(df, structure)
    else:
        print(f"📊 Self-matching with {len(structure['clean_columns'])} fields")
        return convert_testing_format(df, structure)


def analyze_ditto_results(jsonl_path: str) -> Dict[str, Any]:
    """Analyze DITTO matching results."""
    results = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    if not results:
        return {"error": "No results found"}
    
    # Calculate statistics
    total_pairs = len(results)
    matches = sum(1 for r in results if r.get('match', False))
    match_scores = [r.get('match_confidence', 0.0) for r in results]
    
    return {
        'total_pairs': total_pairs,
        'matches': matches,
        'non_matches': total_pairs - matches,
        'match_rate': matches / total_pairs if total_pairs > 0 else 0,
        'avg_score': np.mean(match_scores) if match_scores else 0,
        'score_distribution': match_scores
    }


def extract_hive_data(hive_config: Dict, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Extract data from Hive table."""
    print(f"🔄 Extracting data from: {table_name}")
    
    hive_conn = HiveConnector(hive_config)
    
    if not hive_conn.connect():
        raise RuntimeError("Failed to connect to Hive")
    
    try:
        query = f"SELECT * FROM {table_name}"
        if limit:
            query += f" LIMIT {limit}"
        
        df = hive_conn.execute_query(query)
        
        print(f"✅ Extracted {len(df)} records")
        print(f"📊 Columns: {list(df.columns)}")
        
        return df
        
    finally:
        hive_conn.close()


def run_ditto_matching(input_jsonl: str, output_jsonl: str, ditto_config: Dict) -> bool:
    """Run DITTO matching on the input pairs."""
    print("🔄 Running DITTO entity matching...")
    
    # Build matcher command
    cmd = [
        'python', 'matcher.py',
        '--task', ditto_config['task'],
        '--input_path', input_jsonl,
        '--output_path', output_jsonl,
        '--lm', ditto_config['lm'],
        '--max_len', str(ditto_config['max_len']),
        '--checkpoint_path', ditto_config['checkpoint_path']
    ]
    
    if ditto_config.get('use_gpu', False):
        cmd.append('--use_gpu')
        
    if ditto_config.get('fp16', False):
        cmd.append('--fp16')
    
    print(f"🚀 Command: {' '.join(cmd)}")
    
    # Set environment for GPU
    env = os.environ.copy()
    if ditto_config.get('use_gpu', False):
        env['CUDA_VISIBLE_DEVICES'] = '0'
    
    try:
        # Run the matcher
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print("✅ DITTO matching completed successfully!")
            print("\n📋 Output:")
            print(result.stdout)
            return True
        else:
            print("❌ DITTO matching failed!")
            print("\n❌ Error:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ DITTO matching timed out")
        return False
    except Exception as e:
        print(f"❌ Error running DITTO: {e}")
        return False


def save_results_to_hive(results_jsonl: str, hive_config: Dict, output_table: str) -> bool:
    """Save results back to Hive."""
    print(f"🔄 Saving results to Hive table: {output_table}")
    
    try:
        # Read results
        results = []
        with open(results_jsonl, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        
        print(f"📋 Loaded {len(results)} results")
        
        # Convert to DataFrame for easier handling
        results_df = pd.DataFrame([
            {
                'record_id': r.get('id', 0),
                'left_record': r.get('left', ''),
                'right_record': r.get('right', ''),
                'match_probability': r.get('match_confidence', 0.0),
                'is_match': r.get('match', False),
                'created_at': datetime.now().isoformat()
            }
            for r in results
        ])
        
        print("\n📊 Results DataFrame:")
        print(results_df.head())
        
        # Connect to Hive for saving
        hive_conn = HiveConnector(hive_config)
        if not hive_conn.connect():
            raise RuntimeError("Failed to connect to Hive for saving results")
        
        cursor = hive_conn.connection.cursor()
        
        # Create table if doesn't exist
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {output_table} (
            record_id INT,
            left_record STRING,
            right_record STRING,
            match_probability DOUBLE,
            is_match BOOLEAN,
            created_at STRING
        )
        STORED AS PARQUET
        """
        
        cursor.execute(create_table_sql)
        print(f"✅ Table {output_table} created/verified")
        
        # Insert results (for large datasets, consider bulk loading)
        insert_count = 0
        for _, row in results_df.iterrows():
            # Escape single quotes in strings
            left_record = str(row['left_record']).replace("'", "''")[:1000]  # Truncate long records
            right_record = str(row['right_record']).replace("'", "''")[:1000]
            
            insert_sql = f"""
            INSERT INTO {output_table} VALUES (
                {row['record_id']},
                '{left_record}',
                '{right_record}',
                {row['match_probability']},
                {'true' if row['is_match'] else 'false'},
                '{row['created_at']}'
            )
            """
            
            cursor.execute(insert_sql)
            insert_count += 1
            
            if insert_count % 10 == 0:
                print(f"  Inserted {insert_count}/{len(results_df)} records...")
        
        print(f"✅ Successfully saved {insert_count} results to {output_table}")
        hive_conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error saving to Hive: {e}")
        return False


def main():
    """Main function to run the complete pipeline."""
    parser = argparse.ArgumentParser(description="Hive-DITTO Entity Matching Pipeline")
    
    # Hive connection
    parser.add_argument("--hive-host", required=True, help="Hive server host")
    parser.add_argument("--hive-port", type=int, default=10000, help="Hive server port")
    parser.add_argument("--hive-user", default="hive", help="Hive username")
    parser.add_argument("--hive-database", default="default", help="Hive database")
    parser.add_argument("--hive-auth", default="NOSASL", help="Hive auth method")
    
    # Input/Output
    parser.add_argument("--input-table", required=True, help="Input Hive table")
    parser.add_argument("--output-table", help="Output Hive table (optional)")
    parser.add_argument("--temp-dir", default="./temp", help="Temporary directory")
    parser.add_argument("--sample-size", type=int, help="Sample size for testing")
    
    # DITTO configuration
    parser.add_argument("--task", default="person_records", help="DITTO task")
    parser.add_argument("--lm", default="bert", help="Language model")
    parser.add_argument("--max-len", type=int, default=64, help="Max sequence length")
    parser.add_argument("--checkpoint-path", default="checkpoints/", help="Checkpoint path")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU")
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    parser.add_argument("--matching-mode", default="auto", choices=["auto", "production", "testing"], 
                       help="Matching mode: auto (detect), production (_left/_right columns), testing (self-match)")
    
    args = parser.parse_args()
    
    # Configuration
    hive_config = {
        'host': args.hive_host,
        'port': args.hive_port,
        'database': args.hive_database,
        'username': args.hive_user,
        'auth': args.hive_auth
    }
    
    ditto_config = {
        'task': args.task,
        'lm': args.lm,
        'max_len': args.max_len,
        'checkpoint_path': args.checkpoint_path,
        'use_gpu': args.use_gpu,
        'fp16': args.fp16
    }
    
    # Create temp directory
    os.makedirs(args.temp_dir, exist_ok=True)
    
    input_jsonl = os.path.join(args.temp_dir, 'input_data.jsonl')
    output_jsonl = os.path.join(args.temp_dir, 'output_results.jsonl')
    
    print("📋 Configuration:")
    print(f"  Hive: {hive_config['host']}:{hive_config['port']}")
    print(f"  Input table: {args.input_table}")
    print(f"  Output table: {args.output_table}")
    print(f"  DITTO model: {ditto_config['lm']} ({ditto_config['task']})")
    
    try:
        # Step 1: Extract data from Hive
        df = extract_hive_data(hive_config, args.input_table, args.sample_size)
        
        # Step 2: Convert to DITTO format
        print("🔄 Converting data to DITTO format...")
        ditto_records = convert_to_ditto_format(df, args.matching_mode)
        
        # Save to JSONL file in the format expected by matcher.py
        with open(input_jsonl, 'w', encoding='utf-8') as f:
            for record in ditto_records:
                # Convert from our format to matcher.py expected format
                matcher_record = [record['left'], record['right']]
                f.write(json.dumps(matcher_record, ensure_ascii=False) + '\n')
        
        print(f"💾 Saved to: {input_jsonl}")
        
        # Step 3: Run DITTO matching
        success = run_ditto_matching(input_jsonl, output_jsonl, ditto_config)
        
        if not success:
            print("❌ Pipeline failed at DITTO matching step")
            return 1
        
        # Step 4: Analyze results
        if os.path.exists(output_jsonl):
            print("📊 Analyzing DITTO matching results...")
            stats = analyze_ditto_results(output_jsonl)
            
            if 'error' not in stats:
                print("\n✅ Results Summary:")
                print(f"  Total pairs processed: {stats['total_pairs']}")
                print(f"  Matches found: {stats['matches']}")
                print(f"  Non-matches: {stats['non_matches']}")
                print(f"  Match rate: {stats['match_rate']:.2%}")
                print(f"  Average confidence: {stats['avg_score']:.3f}")
        
        # Step 5: Save to Hive (optional)
        if args.output_table:
            success = save_results_to_hive(output_jsonl, hive_config, args.output_table)
            if not success:
                print("⚠️  Failed to save results to Hive, but matching completed successfully")
        
        print("\n🎉 Pipeline completed successfully!")
        print(f"\n📁 Generated Files:")
        print(f"  ✅ {input_jsonl}")
        print(f"  ✅ {output_jsonl}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())