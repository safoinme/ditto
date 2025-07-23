from datetime import datetime
from kfp import compiler, dsl
from typing import NamedTuple, Optional
import os
import argparse
from kfp.components import create_component_from_func
from kubernetes.client.models import V1EnvVar
from kubernetes import client as k8s_client

CACHE_ENABLED = True

def safe_table_name(table: str) -> str:
    """Return the table part of a fully-qualified hive table name."""
    return table.split('.')[-1]

def extract_and_process_ditto_func(
    hive_host: str,
    hive_port: int,
    hive_user: str,
    hive_database: str,
    input_table: str,
    sample_limit: Optional[int] = None,
    matching_mode: str = 'auto',
    # Ditto parameters
    model_task: str = "person_records",
    checkpoint_path: str = "/checkpoints/",
    lm: str = "bert",
    max_len: int = 64,
    use_gpu: bool = True,
    fp16: bool = True,
    summarize: bool = False
) -> dict:
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
            database=hive_database,
            auth='NOSASL'
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
            # Build matcher command with GPU support
            cmd = [
                "python", "/app/ditto/matcher.py",
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
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            print(f"DITTO completed successfully")
            if result.stdout:
                print(f"DITTO Output: {result.stdout}")
            
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
    processing_results: dict,
    hive_host: str,
    hive_port: int,
    hive_user: str,
    hive_database: str,
    output_table: str,
    save_results: bool = True
) -> str:
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

# Create Kubeflow components
extract_and_process_ditto_op = create_component_from_func(
    func=extract_and_process_ditto_func,
    base_image='172.17.232.16:9001/ditto:1.5',  # Ensure this image has CUDA support
)

save_results_to_hive_op = create_component_from_func(
    func=save_results_to_hive_func,
    base_image='172.17.232.16:9001/ditto:1.5',
)

@dsl.pipeline(
    name="ditto-entity-matching-gpu",
    description="Ditto Entity Matching Pipeline with proper GPU support"
)
def ditto_entity_matching_pipeline_gpu(
    # Hive connection parameters
    hive_host: str = "172.17.235.21",
    hive_port: int = 10000,
    hive_user: str = "hive",
    hive_database: str = "default",
    
    # Input table
    input_table: str = "model_reference",
    
    # Data limits (for testing)
    sample_limit: Optional[int] = None,
    
    # Matching mode
    matching_mode: str = 'auto',
    
    # Ditto model parameters
    model_task: str = "person_records",
    checkpoint_path: str = "/checkpoints",
    lm: str = "bert",
    max_len: int = 64,
    use_gpu: bool = True,
    fp16: bool = True,
    summarize: bool = False,
    
    # Output parameters
    save_to_hive: bool = False,
    output_table: str = "ditto_matching_results"
):
    """
    GPU-enabled Ditto matching pipeline that:
    1. Extracts from Hive → processes with DITTO using GPU → returns results
    2. Optionally saves results back to Hive
    """
    
    # Define environment variables for Hive connectivity
    env_vars = [
        V1EnvVar(name='HIVE_HOST', value=hive_host),
        V1EnvVar(name='HIVE_PORT', value=str(hive_port)),
        V1EnvVar(name='HIVE_USER', value=hive_user),
        V1EnvVar(name='HIVE_DATABASE', value=hive_database),
        # GPU-specific environment variables
        V1EnvVar(name='CUDA_VISIBLE_DEVICES', value='0'),
        V1EnvVar(name='CUDA_DEVICE_ORDER', value='PCI_BUS_ID'),
        V1EnvVar(name='NVIDIA_VISIBLE_DEVICES', value='all'),
        V1EnvVar(name='NVIDIA_DRIVER_CAPABILITIES', value='compute,utility')
    ]
    
    # Step 1: Extract from Hive and process with DITTO using GPU
    process_results = extract_and_process_ditto_op(
        hive_host=hive_host,
        hive_port=hive_port,
        hive_user=hive_user,
        hive_database=hive_database,
        input_table=input_table,
        sample_limit=sample_limit,
        matching_mode=matching_mode,
        model_task=model_task,
        checkpoint_path=checkpoint_path,
        lm=lm,
        max_len=max_len,
        use_gpu=use_gpu,
        fp16=fp16,
        summarize=summarize
    )
    
    # Add environment variables and GPU resources
    for env_var in env_vars:
        process_results.add_env_variable(env_var)
    
    process_results.set_display_name('Extract from Hive and Run DITTO with GPU')
    
    # Proper GPU resource configuration
    if use_gpu:
        # Request GPU resources using the correct Kubernetes resource type
        process_results.container.add_resource_request('nvidia.com/gpu', '1')
        process_results.container.add_resource_limit('nvidia.com/gpu', '1')
        
        # Add node selector to ensure GPU node
        process_results.add_node_selector_constraint('accelerator', 'nvidia-tesla-k80')  # Adjust to your GPU type
        
        # Add tolerations for GPU nodes (if needed)
        from kubernetes.client.models import V1Toleration
        gpu_toleration = V1Toleration(
            key="nvidia.com/gpu",
            operator="Equal",
            value="present",
            effect="NoSchedule"
        )
        process_results.add_toleration(gpu_toleration)
        
        # High memory for GPU processing
        process_results.set_memory_limit('32Gi')
        process_results.set_memory_request('16Gi')
    else:
        # CPU-only resources
        process_results.set_memory_limit('16Gi')
        process_results.set_memory_request('8Gi')
    
    process_results.set_cpu_limit('8')
    process_results.set_cpu_request('4')
    process_results.set_caching_options(enable_caching=False)  # Don't cache GPU jobs
    
    # Step 2: Save results to Hive
    save_results = save_results_to_hive_op(
        processing_results=process_results.output,
        hive_host=hive_host,
        hive_port=hive_port,
        hive_user=hive_user,
        hive_database=hive_database,
        output_table=output_table,
        save_results=save_to_hive
    ).after(process_results)
    
    # Add environment variables (CPU-only step)
    for env_var in env_vars[:4]:  # Only Hive env vars, not GPU ones
        save_results.add_env_variable(env_var)
    save_results.set_display_name('Save Results to Hive')
    save_results.set_caching_options(enable_caching=False)

def compile_pipeline(
    input_table: str = "model_reference",
    hive_host: str = "172.17.235.21",
    pipeline_file: str = "ditto-pipeline-gpu.yaml"
):
    """Compile the GPU-enabled Ditto matching pipeline."""
    try:
        compiler.Compiler().compile(
            pipeline_func=ditto_entity_matching_pipeline_gpu,
            package_path=pipeline_file,
            type_check=True
        )
        
        print(f"\nGPU-Enabled Pipeline compiled successfully!")
        print(f"Pipeline file: {os.path.abspath(pipeline_file)}")
        print(f"Input table: {input_table}")
        print(f"Hive Host: {hive_host}")
        print("GPU Support: Properly configured with:")
        print("   - NVIDIA GPU resource requests/limits")
        print("   - GPU node selection")
        print("   - CUDA environment variables")
        print("   - GPU tolerations and constraints")
        
        return pipeline_file
        
    except Exception as e:
        print(f"ERROR compiling pipeline: {str(e)}")
        raise

def main():
    """Command line interface for GPU pipeline compilation."""
    parser = argparse.ArgumentParser(description="Ditto Matching Kubeflow Pipeline (GPU-Enabled)")
    
    parser.add_argument("--compile", action="store_true", help="Compile the pipeline")
    parser.add_argument("--input-table", default="model_reference", help="Input Hive table")
    parser.add_argument("--hive-host", default="172.17.235.21", help="Hive server host")
    parser.add_argument("--output", default="ditto-pipeline-gpu.yaml", help="Output pipeline file")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    
    args = parser.parse_args()
    
    global CACHE_ENABLED
    if args.no_cache:
        CACHE_ENABLED = False
    
    if args.compile:
        pipeline_file = compile_pipeline(
            input_table=args.input_table,
            hive_host=args.hive_host,
            pipeline_file=args.output
        )
        print(f"\nPipeline Steps:")
        print("1. Extract from Hive + Run DITTO with GPU - All processing with GPU acceleration")
        print("2. Save Results to Hive - Store final results with GPU usage tracking")
        print(f"\nUsage: Upload {args.output} to your Kubeflow Pipelines UI")
        print("Make sure your cluster has GPU nodes with NVIDIA drivers installed!")
        return pipeline_file
    else:
        print("Use --compile flag to compile the pipeline")
        return None

if __name__ == "__main__":
    main()