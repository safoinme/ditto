#!/usr/bin/env python3

import argparse
import json
import os
import sys
import tempfile
import subprocess

def create_sample_data(num_pairs=10):
    """Create sample data for testing DITTO matcher."""
    sample_records = [
        ["COL name VAL John Smith COL age VAL 30 COL city VAL New York", 
         "COL name VAL J. Smith COL age VAL 30 COL city VAL NYC"],
        ["COL name VAL Alice Johnson COL age VAL 25 COL city VAL Boston", 
         "COL name VAL Alice J. COL age VAL 25 COL city VAL Boston MA"],
        ["COL name VAL Bob Wilson COL age VAL 35 COL city VAL Chicago", 
         "COL name VAL Robert Wilson COL age VAL 35 COL city VAL Chicago IL"],
        ["COL name VAL Mary Davis COL age VAL 28 COL city VAL Seattle", 
         "COL name VAL M. Davis COL age VAL 28 COL city VAL Seattle WA"],
        ["COL name VAL David Brown COL age VAL 42 COL city VAL Miami", 
         "COL name VAL David Brown COL age VAL 42 COL city VAL Miami FL"],
        ["COL name VAL Sarah Miller COL age VAL 31 COL city VAL Denver", 
         "COL name VAL Sarah M. COL age VAL 31 COL city VAL Denver CO"],
        ["COL name VAL Michael Garcia COL age VAL 29 COL city VAL Phoenix", 
         "COL name VAL Mike Garcia COL age VAL 29 COL city VAL Phoenix AZ"],
        ["COL name VAL Lisa Anderson COL age VAL 33 COL city VAL Portland", 
         "COL name VAL L. Anderson COL age VAL 33 COL city VAL Portland OR"],
        ["COL name VAL James Taylor COL age VAL 27 COL city VAL Austin", 
         "COL name VAL Jim Taylor COL age VAL 27 COL city VAL Austin TX"],
        ["COL name VAL Jennifer White COL age VAL 26 COL city VAL Nashville", 
         "COL name VAL Jen White COL age VAL 26 COL city VAL Nashville TN"],
        # Some non-matching pairs
        ["COL name VAL John Smith COL age VAL 30 COL city VAL New York", 
         "COL name VAL Alice Johnson COL age VAL 25 COL city VAL Boston"],
        ["COL name VAL Bob Wilson COL age VAL 35 COL city VAL Chicago", 
         "COL name VAL Mary Davis COL age VAL 28 COL city VAL Seattle"],
        ["COL name VAL David Brown COL age VAL 42 COL city VAL Miami", 
         "COL name VAL Michael Garcia COL age VAL 29 COL city VAL Phoenix"],
    ]
    
    # Return requested number of pairs
    return sample_records[:min(num_pairs, len(sample_records))]

def load_data_from_file(file_path):
    """Load data from JSONL file."""
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line.strip())
                records.append(record)
    return records

def test_matcher(
    input_data=None,
    input_file=None,
    num_sample_pairs=10,
    model_task="person_records",
    checkpoint_path="/checkpoints",
    lm="bert",
    max_len=64,
    use_gpu=True,
    fp16=True,
    summarize=False,
    output_file=None
):
    """Test DITTO matcher with sample or provided data."""
    
    try:
        # GPU Detection and Setup
        if use_gpu:
            print("=== GPU SETUP ===")
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

        # Get test data
        if input_file:
            print(f"=== Loading data from {input_file} ===")
            ditto_records = load_data_from_file(input_file)
        elif input_data:
            print("=== Using provided data ===")
            ditto_records = input_data
        else:
            print(f"=== Creating {num_sample_pairs} sample record pairs ===")
            ditto_records = create_sample_data(num_sample_pairs)
        
        print(f"Testing with {len(ditto_records)} record pairs")

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
                print("Available files:")
                for root, dirs, files in os.walk('.'):
                    for file in files:
                        if file.endswith('.py'):
                            print(f"  {os.path.join(root, file)}")
                return {
                    "results": [],
                    "metrics": {"total_pairs": 0, "matches": 0, "non_matches": 0},
                    "gpu_used": False,
                    "status": "error: matcher.py not found"
                }

            # Build matcher command
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

            if fp16 and use_gpu:
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

            # Save results to file if requested
            if output_file:
                final_results = {
                    "results": results,
                    "metrics": metrics,
                    "gpu_used": use_gpu,
                    "status": "success",
                    "input_records": ditto_records
                }
                with open(output_file, 'w') as f:
                    json.dump(final_results, f, indent=2)
                print(f"Results saved to {output_file}")

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
        print(f"ERROR in test_matcher: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "results": [],
            "metrics": {"total_pairs": 0, "matches": 0, "non_matches": 0},
            "gpu_used": False,
            "status": f"error: {str(e)}"
        }


def main():
    parser = argparse.ArgumentParser(description='DITTO Matcher Test - No Hive Required')
    
    # Input options
    parser.add_argument('--input-file', help='JSONL file with test data (format: [["left", "right"], ...])')
    parser.add_argument('--num-sample-pairs', type=int, default=10, help='Number of sample pairs to generate if no input file')
    
    # Model parameters
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
    
    # Output options
    parser.add_argument('--output-file', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    print("=== DITTO Matcher Test ===")
    print(f"Configuration:")
    print(f"  Input file: {args.input_file or 'Generated sample data'}")
    print(f"  Sample pairs: {args.num_sample_pairs if not args.input_file else 'N/A'}")
    print(f"  GPU: {args.use_gpu}, FP16: {args.fp16}")
    print(f"  Model: {args.model_task} ({args.lm})")
    print(f"  Checkpoints: {args.checkpoint_path}")
    print()
    
    # Run the test
    results = test_matcher(
        input_file=args.input_file,
        num_sample_pairs=args.num_sample_pairs,
        model_task=args.model_task,
        checkpoint_path=args.checkpoint_path,
        lm=args.lm,
        max_len=args.max_len,
        use_gpu=args.use_gpu,
        fp16=args.fp16,
        summarize=args.summarize,
        output_file=args.output_file
    )
    
    print(f"\n=== Results ===")
    print(f"Status: {results.get('status')}")
    print(f"Metrics: {results.get('metrics')}")
    print(f"GPU Used: {results.get('gpu_used')}")
    
    if results.get('status') == 'success':
        metrics = results.get('metrics', {})
        total = metrics.get('total_pairs', 0)
        matches = metrics.get('matches', 0)
        if total > 0:
            match_rate = (matches / total) * 100
            print(f"Match Rate: {match_rate:.1f}% ({matches}/{total})")
        
        print("\n✅ Matcher test completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Matcher test failed: {results.get('status')}")
        sys.exit(1)


if __name__ == "__main__":
    main() 