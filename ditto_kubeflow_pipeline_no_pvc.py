from datetime import datetime
from kfp import compiler, dsl
from typing import NamedTuple, Optional
import os
import argparse
from kfp.components import create_component_from_func
from kubernetes.client.models import V1EnvVar
import json
import time

CACHE_ENABLED = True

def safe_table_name(table: str) -> str:
    """Return the table part of a fully-qualified hive table name."""
    return table.split('.')[-1]

def extract_hive_data_func(
    hive_host: str,
    hive_port: int,
    hive_user: str,
    hive_database: str,
    input_table: str,
    sample_limit: Optional[int] = None,
    matching_mode: str = 'auto'
) -> str:
    """Extract data from Hive table and return DITTO format data as string."""
    from pyhive import hive
    import pandas as pd
    import json
    
    try:
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
        print(f"Columns: {list(df.columns)}")
        
        # Detect and convert to DITTO format (same logic as before)
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
                
                record = {
                    "left": left_text,
                    "right": right_text,
                    "id": idx
                }
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
                
                record = {
                    "left": record_text,
                    "right": record_text,
                    "id": idx
                }
                records.append(record)
            
            return records
        
        def convert_to_ditto_format(df, matching_mode):
            structure = detect_table_structure(df)
            
            # Override detection if mode is specified
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
        
        # Return data as JSONL string instead of writing to file
        jsonl_lines = []
        for record in ditto_records:
            # Convert to matcher.py expected format: [left_record, right_record]
            matcher_record = [record['left'], record['right']]
            jsonl_lines.append(json.dumps(matcher_record, ensure_ascii=True))
        
        jsonl_data = '\n'.join(jsonl_lines)
        print(f"Generated {len(jsonl_lines)} JSONL records")
        
        connection.close()
        return jsonl_data
        
    except Exception as e:
        print(f"Error in extract_hive_data_func: {str(e)}")
        raise

def run_ditto_matching_func(
    input_data: str,
    model_task: str = "person_records",
    checkpoint_path: str = "/checkpoints/",
    lm: str = "bert",
    max_len: int = 64,
    use_gpu: bool = True,
    fp16: bool = True,
    summarize: bool = False
) -> NamedTuple('Outputs', [('results_data', str), ('metrics', dict)]):
    """Run ditto matching on the input pairs data."""
    import subprocess
    import os
    import json
    import jsonlines
    import tempfile
    from collections import namedtuple
    
    # Write input data to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as temp_input:
        temp_input.write(input_data)
        input_path = temp_input.name
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as temp_output:
        output_path = temp_output.name
    
    try:
        # Build matcher command
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
        if fp16:
            cmd.append("--fp16")
        if summarize:
            cmd.append("--summarize")
        
        # Run matcher
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Matcher stdout: {result.stdout}")
        
        # Read results and return as string
        results_data = ""
        metrics = {"total_pairs": 0, "matches": 0, "non_matches": 0}
        
        if os.path.exists(output_path):
            with jsonlines.open(output_path) as reader:
                results_lines = []
                for line in reader:
                    results_lines.append(json.dumps(line))
                    metrics["total_pairs"] += 1
                    if line.get('match', 0) == 1:
                        metrics["matches"] += 1
                    else:
                        metrics["non_matches"] += 1
                
                results_data = '\n'.join(results_lines)
        
        print(f"Matching completed. Metrics: {metrics}")
        
        output = namedtuple('Outputs', ['results_data', 'metrics'])
        return output(results_data, metrics)
        
    except subprocess.CalledProcessError as e:
        print(f"Matcher failed with error: {e}")
        print(f"Stderr: {e.stderr}")
        raise
    except Exception as e:
        print(f"Error in run_ditto_matching_func: {str(e)}")
        raise
    finally:
        # Clean up temporary files
        for temp_file in [input_path, output_path]:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

def save_results_to_hive_func(
    results_data: str,
    hive_host: str,
    hive_port: int,
    hive_user: str,
    hive_database: str,
    output_table: str,
    save_results: bool = True
) -> str:
    """Save matching results to Hive from results data string."""
    if not save_results:
        print("Skipping Hive save as save_results is False")
        return "Skipped"
    
    from pyhive import hive
    import pandas as pd
    import json
    import tempfile
    import os
    
    try:
        # Parse results data
        results = []
        for line in results_data.strip().split('\n'):
            if line.strip():
                result = json.loads(line)
                results.append({
                    'left_record': json.dumps(result.get('left', {})),
                    'right_record': json.dumps(result.get('right', {})),
                    'match': result.get('match', 0),
                    'match_confidence': result.get('match_confidence', 0.0),
                    'processing_timestamp': datetime.now().isoformat()
                })
        
        if not results:
            print("No results to save")
            return "No results"
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
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
            processing_timestamp STRING
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
            
            print(f"Successfully saved {len(results)} results to {output_table}")
            return f"Saved {len(results)} results to {output_table}"
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
        connection.close()
        
    except Exception as e:
        print(f"Error saving to Hive: {str(e)}")
        return f"Error: {str(e)}"

# Create Kubeflow components
extract_hive_data_op = create_component_from_func(
    func=extract_hive_data_func,
    base_image='172.17.232.16:9001/ditto:1.5',
)

run_ditto_matching_op = create_component_from_func(
    func=run_ditto_matching_func,
    base_image='172.17.232.16:9001/ditto:1.5',
)

save_results_to_hive_op = create_component_from_func(
    func=save_results_to_hive_func,
    base_image='172.17.232.16:9001/ditto:1.5',
)

def generate_pipeline_name(input_table: str) -> str:
    """Generate a unique pipeline name based on table and timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_safe = safe_table_name(input_table).replace('_', '-')
    return f"ditto-matching-{table_safe}-{timestamp}"

@dsl.pipeline(
    name="ditto-entity-matching-no-pvc",
    description="Ditto Entity Matching Pipeline without PVC - uses step outputs"
)
def ditto_entity_matching_pipeline_no_pvc(
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
    Complete Ditto matching pipeline without PVC that:
    1. Extracts data from Hive table and returns as string
    2. Runs ditto matching on string data
    3. Optionally saves results back to Hive
    """
    
    # Define environment variables for Hive connectivity
    env_vars = [
        V1EnvVar(name='HIVE_HOST', value=hive_host),
        V1EnvVar(name='HIVE_PORT', value=str(hive_port)),
        V1EnvVar(name='HIVE_USER', value=hive_user),
        V1EnvVar(name='HIVE_DATABASE', value=hive_database)
    ]
    
    # Step 1: Extract data from Hive table
    extract_data = extract_hive_data_op(
        hive_host=hive_host,
        hive_port=hive_port,
        hive_user=hive_user,
        hive_database=hive_database,
        input_table=input_table,
        sample_limit=sample_limit,
        matching_mode=matching_mode
    )
    
    # Add environment variables (no volume needed)
    for env_var in env_vars:
        extract_data.add_env_variable(env_var)
    extract_data.set_display_name('Extract Data from Hive')
    extract_data.set_caching_options(enable_caching=CACHE_ENABLED)
    
    # Step 2: Run ditto matching using output from step 1
    matching_results = run_ditto_matching_op(
        input_data=extract_data.output,  # Use output from previous step
        model_task=model_task,
        checkpoint_path=checkpoint_path,
        lm=lm,
        max_len=max_len,
        use_gpu=use_gpu,
        fp16=fp16,
        summarize=summarize
    ).after(extract_data)
    
    # Add GPU resources (no volume needed)
    matching_results.set_display_name('Run DITTO Matching')  
    matching_results.set_gpu_limit(1)
    matching_results.set_memory_limit('8Gi')
    matching_results.set_cpu_limit('4')
    matching_results.set_caching_options(enable_caching=False)  # Don't cache matching results
    
    # Step 3: Optionally save results to Hive using output from step 2
    save_results = save_results_to_hive_op(
        results_data=matching_results.outputs['results_data'],  # Use output from previous step
        hive_host=hive_host,
        hive_port=hive_port,
        hive_user=hive_user,
        hive_database=hive_database,
        output_table=output_table,
        save_results=save_to_hive
    ).after(matching_results)
    
    # Add environment variables (no volume needed)
    for env_var in env_vars:
        save_results.add_env_variable(env_var)
    save_results.set_display_name('Save Results to Hive')
    save_results.set_caching_options(enable_caching=False)

def compile_pipeline(
    input_table: str = "model_reference",
    hive_host: str = "172.17.235.21",
    pipeline_file: str = "ditto-pipeline-no-pvc.yaml"
):
    """Compile the Ditto matching pipeline without PVC."""
    try:
        compiler.Compiler().compile(
            pipeline_func=ditto_entity_matching_pipeline_no_pvc,
            package_path=pipeline_file,
            type_check=True
        )
        
        pipeline_name = generate_pipeline_name(input_table)
        print(f"\nPipeline '{pipeline_name}' compiled successfully!")
        print(f"Pipeline file: {os.path.abspath(pipeline_file)}")
        print(f"Input table: {input_table}")
        print(f"Hive Host: {hive_host}")
        print("No PVC required - uses step outputs for data passing")
        
        return pipeline_file
        
    except Exception as e:
        print(f"Error compiling pipeline: {str(e)}")
        raise

def main():
    """Command line interface for pipeline compilation."""
    parser = argparse.ArgumentParser(description="Ditto Matching Kubeflow Pipeline (No PVC)")
    
    # Action flags
    parser.add_argument("--compile", action="store_true", help="Compile the pipeline")
    
    # Pipeline parameters (optional with defaults)
    parser.add_argument("--input-table", default="model_reference", 
                       help="Input Hive table")
    parser.add_argument("--hive-host", default="172.17.235.21", help="Hive server host")
    parser.add_argument("--output", default="ditto-pipeline-no-pvc.yaml", help="Output pipeline file")
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
        print("1. Extract Data from Hive - Extract and return data as string output")
        print("2. Run DITTO Matching - Process input string data, return results as string")
        print("3. Save Results to Hive - Process results string and save to Hive")
        print(f"\nUsage: Upload {args.output} to your Kubeflow Pipelines UI")
        return pipeline_file
    else:
        print("Use --compile flag to compile the pipeline")
        print("Example: python ditto_kubeflow_pipeline_no_pvc.py --compile")
        print("Example: python ditto_kubeflow_pipeline_no_pvc.py --compile --input-table your_table --hive-host your_host")
        return None

if __name__ == "__main__":
    main()