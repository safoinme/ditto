#!/usr/bin/env python3

import argparse
import json
import os
import sys
from datetime import datetime

def extract_and_process_ditto_func(
    hive_host,
    hive_port,
    hive_user,
    hive_database,
    input_table,
    sample_limit = None,
    matching_mode = 'auto',
    # Ditto parameters
    model_task = "person_records",
    checkpoint_path = "/checkpoints/",
    lm = "bert",
    max_len = 64,
    use_gpu = True,
    fp16 = True,
    summarize = False
):
    """Extract from Hive, process with DITTO using GPU, all in one step."""
    from pyhive import hive
    import pandas as pd
    import json
    import subprocess
    import tempfile
    import os

    try:
        # GPU Detection and Setup
        if use_gpu:
            print("=== GPU SETUP ===")
            # Check if GPU is available
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    current_device = torch.cuda.current_device()
                    gpu_name = torch.cuda.get_device_name(current_device)
                    print(f"GPU Available: {gpu_name}")
                    print(f"GPU Count: {gpu_count}")
                    print(f"Current Device: {current_device}")

                    # Set CUDA environment variables
                    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
                else:
                    print("WARNING: GPU requested but not available, falling back to CPU")
                    use_gpu = False
            except ImportError:
                print("WARNING: PyTorch not available, falling back to CPU")
                use_gpu = False
        else:
            print("=== CPU MODE ===")

        print("=== STEP 1: Extract from Hive ===")
        # Connect to Hive
        connection = hive.Connection(
            host=hive_host,
            port=hive_port,
            username=hive_user,
            database=hive_database
        )

        # Extract data
        query = f"SELECT * FROM {input_table}"
        if sample_limit:
            query += f" LIMIT {sample_limit}"

        df = pd.read_sql(query, connection)
        print(f"Extracted {len(df)} records from {input_table}")
        connection.close()

        print("=== STEP 2: Convert to DITTO format ===")
        # Convert to DITTO format (reusing existing logic)
        def detect_table_structure(df):
            columns = list(df.columns)
            clean_columns = []
            for col in columns:
                if '.' in col:
                    clean_col = col.split('.', 1)[1]
                else:
                    clean_col = col
                clean_columns.append(clean_col)

            left_columns = [col for col in clean_columns if col.endswith('_left')]
            right_columns = [col for col in clean_columns if col.endswith('_right')]
            left_fields = {col[:-5] for col in left_columns}
            right_fields = {col[:-6] for col in right_columns}
            matching_fields = left_fields.intersection(right_fields)

            if matching_fields:
                structure_type = "production"
                message = f"Production table detected with {len(matching_fields)} matching field pairs"
            else:
                structure_type = "testing"
                message = f"Testing table detected with {len(clean_columns)} fields for self-matching"

            return {
                'type': structure_type,
                'columns': columns,
                'clean_columns': clean_columns,
                'matching_fields': list(matching_fields),
                'message': message
            }

        def convert_production_format(df, structure):
            records = []
            for idx, row in df.iterrows():
                left_parts = []
                right_parts = []

                for field in structure['matching_fields']:
                    left_col = None
                    right_col = None

                    for col in df.columns:
                        clean_col = col.split('.', 1)[1] if '.' in col else col
                        if clean_col == f"{field}_left":
                            left_col = col
                        elif clean_col == f"{field}_right":
                            right_col = col

                    if left_col and pd.notna(row[left_col]) and str(row[left_col]).strip():
                        value = str(row[left_col]).strip()
                        left_parts.append(f"COL {field} VAL {value}")

                    if right_col and pd.notna(row[right_col]) and str(row[right_col]).strip():
                        value = str(row[right_col]).strip()
                        right_parts.append(f"COL {field} VAL {value}")

                left_text = " ".join(left_parts)
                right_text = " ".join(right_parts)

                record = [left_text, right_text]  # Direct DITTO format
                records.append(record)

            return records

        def convert_testing_format(df, structure):
            records = []
            for idx, row in df.iterrows():
                col_val_parts = []

                for col in df.columns:
                    if pd.notna(row[col]) and str(row[col]).strip():
                        if '.' in col:
                            clean_col = col.split('.', 1)[1]
                        else:
                            clean_col = col

                        value = str(row[col]).strip()
                        col_val_parts.append(f"COL {clean_col} VAL {value}")

                record_text = " ".join(col_val_parts)
                record = [record_text, record_text]  # Direct DITTO format
                records.append(record)

            return records

        def convert_to_ditto_format(df, matching_mode):
            structure = detect_table_structure(df)

            if matching_mode == 'production':
                structure['type'] = 'production'
            elif matching_mode == 'testing':
                structure['type'] = 'testing'

            print(structure['message'])

            if structure['type'] == 'production':
                if not structure['matching_fields']:
                    raise ValueError("Production mode requires _left/_right column pairs, but none were found!")
                return convert_production_format(df, structure)
            else:
                return convert_testing_format(df, structure)

        ditto_records = convert_to_ditto_format(df, matching_mode)
        print(f"Converted {len(ditto_records)} records to DITTO format")

        print("=== STEP 3: Run DITTO Matching with GPU ===")
        # Write to temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as temp_input:
            for record in ditto_records:
                temp_input.write(json.dumps(record, ensure_ascii=True) + '\n')
            input_path = temp_input.name

        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as temp_output:
            output_path = temp_output.name

        try:
            # Debug: Check current directory and files
            print(f"Current working directory: {os.getcwd()}")
            print(f"Files in current directory: {os.listdir('.')}")
            print(f"Checkpoint path exists: {os.path.exists(checkpoint_path)}")
            if os.path.exists(checkpoint_path):
                print(f"Files in checkpoint path: {os.listdir(checkpoint_path)}")

            # Check for matcher.py in different locations
            matcher_locations = [
                "/home/jovyan/matcher.py",
                "./matcher.py",
                "/app/matcher.py"
            ]
            matcher_path = None
            for loc in matcher_locations:
                if os.path.exists(loc):
                    matcher_path = loc
                    print(f"Found matcher.py at: {loc}")
                    break

            if not matcher_path:
                print("ERROR: matcher.py not found in any expected location")
                return {
                    "results": [],
                    "metrics": {"total_pairs": 0, "matches": 0, "non_matches": 0},
                    "gpu_used": False,
                    "status": "error: matcher.py not found"
                }

            # Build matcher command with GPU support
            cmd = [
                "python", matcher_path,
                "--task", model_task,
                "--input_path", input_path,
                "--output_path", output_path,
                "--lm", lm,
                "--max_len", str(max_len),
                "--checkpoint_path", checkpoint_path
            ]

            if use_gpu:
                cmd.append("--use_gpu")
                print("Running DITTO with GPU acceleration")
            else:
                print("Running DITTO with CPU")

            if fp16 and use_gpu:  # FP16 only makes sense with GPU
                cmd.append("--fp16")
                print("Using FP16 mixed precision")

            if summarize:
                cmd.append("--summarize")

            # Set GPU environment for the subprocess
            env = os.environ.copy()
            if use_gpu:
                env['CUDA_VISIBLE_DEVICES'] = '0'
                env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

            print(f"Running DITTO command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)

            if result.returncode != 0:
                print(f"DITTO command failed with exit code {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")

                # Try without GPU as fallback
                if use_gpu:
                    print("Retrying without GPU...")
                    cmd_cpu = [c for c in cmd if c not in ['--use_gpu', '--fp16']]
                    result = subprocess.run(cmd_cpu, capture_output=True, text=True, env=env)
                    if result.returncode != 0:
                        raise subprocess.CalledProcessError(result.returncode, cmd_cpu, result.stdout, result.stderr)
                    use_gpu = False
                else:
                    raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

            print(f"DITTO completed successfully")
            if result.stdout:
                print(f"DITTO Output: {result.stdout}")
            if result.stderr:
                print(f"DITTO Warnings: {result.stderr}")

            # Read and process results
            results = []
            metrics = {"total_pairs": 0, "matches": 0, "non_matches": 0}

            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            result_data = json.loads(line.strip())
                            results.append(result_data)
                            metrics["total_pairs"] += 1
                            if result_data.get('match', 0) == 1:
                                metrics["matches"] += 1
                            else:
                                metrics["non_matches"] += 1

            print(f"Processing completed. Metrics: {metrics}")

            return {
                "results": results,
                "metrics": metrics,
                "gpu_used": use_gpu,
                "status": "success"
            }

        finally:
            # Clean up temporary files
            for temp_file in [input_path, output_path]:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    except Exception as e:
        print(f"ERROR in extract_and_process_ditto_func: {str(e)}")
        return {
            "results": [],
            "metrics": {"total_pairs": 0, "matches": 0, "non_matches": 0},
            "gpu_used": False,
            "status": f"error: {str(e)}"
        }


def save_results_to_hive_func(
    processing_results,
    hive_host,
    hive_port,
    hive_user,
    hive_database,
    output_table,
    save_results = True
):
    """Save processing results to Hive."""
    if not save_results:
        print("Skipping Hive save as save_results is False")
        return "Skipped"

    if processing_results.get("status") != "success":
        return f"ERROR: Cannot save results due to processing error: {processing_results.get('status', 'unknown error')}"

    from pyhive import hive
    import pandas as pd
    import json
    import tempfile
    import os

    try:
        results_data = processing_results.get("results", [])
        if not results_data:
            print("No results to save")
            return "No results"

        # Convert results to DataFrame
        processed_results = []
        for result in results_data:
            processed_results.append({
                'left_record': json.dumps(result.get('left', {})),
                'right_record': json.dumps(result.get('right', {})),
                'match': result.get('match', 0),
                'match_confidence': result.get('match_confidence', 0.0),
                'processing_timestamp': datetime.now().isoformat(),
                'gpu_used': processing_results.get('gpu_used', False)
            })

        df = pd.DataFrame(processed_results)

        # Connect to Hive
        connection = hive.Connection(
            host=hive_host,
            port=hive_port,
            username=hive_user,
            database=hive_database
        )

        cursor = connection.cursor()

        # Create table if it doesn't exist
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {output_table} (
            left_record STRING,
            right_record STRING,
            match INT,
            match_confidence FLOAT,
            processing_timestamp STRING,
            gpu_used BOOLEAN
        )
        STORED AS PARQUET
        """

        cursor.execute(create_table_sql)

        # Save DataFrame to temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            df.to_csv(temp_file.name, index=False, header=False)
            temp_path = temp_file.name

        try:
            # Load data into Hive table
            load_sql = f"""
            LOAD DATA LOCAL INPATH '{temp_path}' 
            INTO TABLE {output_table}
            """
            cursor.execute(load_sql)

            gpu_status = "with GPU" if processing_results.get('gpu_used') else "with CPU"
            print(f"Successfully saved {len(processed_results)} results to {output_table} (processed {gpu_status})")
            return f"Saved {len(processed_results)} results to {output_table} (processed {gpu_status})"

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        connection.close()

    except Exception as e:
        print(f"ERROR saving to Hive: {str(e)}")
        return f"Error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='DITTO Entity Matching Pipeline - Docker Runner')
    
    # Hive connection parameters
    parser.add_argument('--hive-host', default='172.17.235.21', help='Hive host')
    parser.add_argument('--hive-port', type=int, default=10000, help='Hive port')
    parser.add_argument('--hive-user', default='lhimer', help='Hive username')
    parser.add_argument('--hive-database', default='preprocessed_analytics', help='Hive database')
    parser.add_argument('--input-table', default='preprocessed_analytics.model_reference', help='Input table')
    parser.add_argument('--output-table', default='ditto_matching_results', help='Output table')
    
    # Processing parameters
    parser.add_argument('--sample-limit', type=int, help='Limit number of records to process')
    parser.add_argument('--matching-mode', default='auto', choices=['auto', 'production', 'testing'], help='Matching mode')
    parser.add_argument('--model-task', default='person_records', help='Model task')
    parser.add_argument('--checkpoint-path', default='/checkpoints', help='Path to model checkpoints')
    parser.add_argument('--lm', default='bert', help='Language model')
    parser.add_argument('--max-len', type=int, default=64, help='Maximum sequence length')
    
    # GPU and performance parameters
    parser.add_argument('--use-gpu', action='store_true', default=True, help='Use GPU acceleration')
    parser.add_argument('--no-gpu', dest='use_gpu', action='store_false', help='Disable GPU acceleration')
    parser.add_argument('--fp16', action='store_true', default=True, help='Use FP16 mixed precision')
    parser.add_argument('--no-fp16', dest='fp16', action='store_false', help='Disable FP16')
    parser.add_argument('--summarize', action='store_true', default=False, help='Enable summarization')
    
    # Output control
    parser.add_argument('--save-to-hive', action='store_true', default=False, help='Save results to Hive')
    parser.add_argument('--output-file', help='Save results to JSON file instead of Hive')
    
    args = parser.parse_args()
    
    print("=== DITTO Entity Matching Pipeline - Docker Runner ===")
    print(f"Configuration:")
    print(f"  Hive: {args.hive_host}:{args.hive_port} ({args.hive_database})")
    print(f"  Input: {args.input_table}")
    print(f"  GPU: {args.use_gpu}, FP16: {args.fp16}")
    print(f"  Model: {args.model_task} ({args.lm})")
    print(f"  Checkpoints: {args.checkpoint_path}")
    print()
    
    # Step 1: Extract and process with DITTO
    print("Starting DITTO processing...")
    results = extract_and_process_ditto_func(
        hive_host=args.hive_host,
        hive_port=args.hive_port,
        hive_user=args.hive_user,
        hive_database=args.hive_database,
        input_table=args.input_table,
        sample_limit=args.sample_limit,
        matching_mode=args.matching_mode,
        model_task=args.model_task,
        checkpoint_path=args.checkpoint_path,
        lm=args.lm,
        max_len=args.max_len,
        use_gpu=args.use_gpu,
        fp16=args.fp16,
        summarize=args.summarize
    )
    
    print(f"\nProcessing Status: {results.get('status')}")
    print(f"Metrics: {results.get('metrics')}")
    print(f"GPU Used: {results.get('gpu_used')}")
    
    # Step 2: Save results
    if args.output_file:
        # Save to JSON file
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")
    elif args.save_to_hive:
        # Save to Hive
        save_status = save_results_to_hive_func(
            processing_results=results,
            hive_host=args.hive_host,
            hive_port=args.hive_port,
            hive_user=args.hive_user,
            hive_database=args.hive_database,
            output_table=args.output_table,
            save_results=True
        )
        print(f"Hive save status: {save_status}")
    else:
        print("Results not saved (use --save-to-hive or --output-file)")
    
    # Return appropriate exit code
    if results.get('status') == 'success':
        print("\n✅ Pipeline completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Pipeline failed: {results.get('status')}")
        sys.exit(1)


if __name__ == "__main__":
    main() 