#!/usr/bin/env python3
"""
Kubeflow pipeline for Hive data extraction, DITTO entity matching, and result storage.
"""

import kfp
from kfp import dsl
from kfp.components import create_component_from_func, OutputPath, InputPath
from typing import NamedTuple

def extract_hive_data_op(
    hive_table: str,
    output_jsonl_path: OutputPath(str),
    hive_host: str = "localhost",
    hive_port: int = 10000,
    hive_database: str = "default"
) -> NamedTuple('Outputs', [('record_count', int), ('output_path', str)]):
    """
    Extract data from Hive table and convert to JSONL format for DITTO.
    
    Args:
        hive_table: Hive table name (e.g., 'base.table')
        output_jsonl_path: Path to save the JSONL output
        hive_host: Hive server host
        hive_port: Hive server port
        hive_database: Hive database name
    
    Returns:
        Tuple containing record count and output path
    """
    from pyhive import hive
    import json
    import pandas as pd
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Connect to Hive
        logger.info(f"Connecting to Hive at {hive_host}:{hive_port}")
        connection = hive.Connection(
            host=hive_host,
            port=hive_port,
            database=hive_database,
            auth='NOSASL'  # Adjust authentication as needed
        )
        
        # Query the table
        query = f"SELECT * FROM {hive_table}"
        logger.info(f"Executing query: {query}")
        
        df = pd.read_sql(query, connection)
        logger.info(f"Retrieved {len(df)} records from {hive_table}")
        
        # Convert to DITTO JSONL format
        jsonl_records = []
        
        for idx, row in df.iterrows():
            # Convert row to DITTO COL/VAL format
            left_parts = []
            right_parts = []
            
            # Split columns between left and right (you can adjust this logic)
            columns = list(df.columns)
            mid_point = len(columns) // 2
            
            # First half as left record
            for col in columns[:mid_point]:
                if pd.notna(row[col]) and str(row[col]).strip():
                    left_parts.append(f"COL {col} VAL {row[col]}")
            
            # Second half as right record  
            for col in columns[mid_point:]:
                if pd.notna(row[col]) and str(row[col]).strip():
                    right_parts.append(f"COL {col} VAL {row[col]}")
            
            # Create JSONL record for matching
            record = {
                "left": " ".join(left_parts),
                "right": " ".join(right_parts),
                "id": idx
            }
            jsonl_records.append(record)
        
        # Write to JSONL file
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            for record in jsonl_records:
                f.write(json.dumps(record) + '\n')
        
        logger.info(f"Saved {len(jsonl_records)} records to {output_jsonl_path}")
        connection.close()
        
        return (len(jsonl_records), output_jsonl_path)
        
    except Exception as e:
        logger.error(f"Error extracting data from Hive: {e}")
        raise

def run_ditto_matching_op(
    input_jsonl_path: InputPath(str),
    output_jsonl_path: OutputPath(str),
    task: str = "person_records",
    lm: str = "bert",
    max_len: int = 64,
    checkpoint_path: str = "checkpoints/"
) -> NamedTuple('Outputs', [('matches_found', int), ('output_path', str)]):
    """
    Run DITTO entity matching on the input data.
    
    Args:
        input_jsonl_path: Path to input JSONL file
        output_jsonl_path: Path to save matching results
        task: DITTO task name
        lm: Language model to use
        max_len: Maximum sequence length
        checkpoint_path: Path to model checkpoints
    
    Returns:
        Tuple containing number of matches found and output path
    """
    import subprocess
    import json
    import os
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Set environment variables for offline model usage
        os.environ['BERT_MODEL_PATH'] = '/models/bert-base-uncased'
        os.environ['ROBERTA_MODEL_PATH'] = '/models/roberta-base'
        os.environ['DISTILBERT_MODEL_PATH'] = '/models/distilbert-base-uncased'
        os.environ['NLTK_DATA'] = '/nltk_data'
        
        # Prepare DITTO command
        cmd = [
            'python', 'matcher.py',
            '--task', task,
            '--input_path', input_jsonl_path,
            '--output_path', output_jsonl_path,
            '--lm', lm,
            '--max_len', str(max_len),
            '--use_gpu',
            '--fp16',
            '--checkpoint_path', checkpoint_path
        ]
        
        # Set CUDA device if available
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        
        logger.info(f"Running DITTO matching: {' '.join(cmd)}")
        
        # Run DITTO matcher
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd='/workspace'  # Adjust working directory as needed
        )
        
        if result.returncode != 0:
            logger.error(f"DITTO matching failed: {result.stderr}")
            raise RuntimeError(f"DITTO matching failed: {result.stderr}")
        
        logger.info(f"DITTO matching completed: {result.stdout}")
        
        # Count matches in output
        matches_found = 0
        if os.path.exists(output_jsonl_path):
            with open(output_jsonl_path, 'r') as f:
                for line in f:
                    record = json.loads(line)
                    if record.get('match', False):  # Adjust based on output format
                        matches_found += 1
        
        logger.info(f"Found {matches_found} matches in output")
        return (matches_found, output_jsonl_path)
        
    except Exception as e:
        logger.error(f"Error running DITTO matching: {e}")
        raise

def save_results_to_hive_op(
    input_jsonl_path: InputPath(str),
    hive_output_table: str,
    hive_host: str = "localhost",
    hive_port: int = 10000,
    hive_database: str = "default"
) -> NamedTuple('Outputs', [('records_saved', int)]):
    """
    Save DITTO matching results back to Hive table.
    
    Args:
        input_jsonl_path: Path to DITTO output JSONL file
        hive_output_table: Hive table name to save results
        hive_host: Hive server host
        hive_port: Hive server port
        hive_database: Hive database name
    
    Returns:
        Number of records saved
    """
    from pyhive import hive
    import json
    import pandas as pd
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Read DITTO results
        results = []
        with open(input_jsonl_path, 'r') as f:
            for line in f:
                results.append(json.loads(line))
        
        logger.info(f"Loaded {len(results)} results from {input_jsonl_path}")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Connect to Hive
        logger.info(f"Connecting to Hive at {hive_host}:{hive_port}")
        connection = hive.Connection(
            host=hive_host,
            port=hive_port,
            database=hive_database,
            auth='NOSASL'
        )
        
        cursor = connection.cursor()
        
        # Create table if it doesn't exist (adjust schema as needed)
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {hive_output_table} (
            id INT,
            left_record STRING,
            right_record STRING,
            match_probability DOUBLE,
            is_match BOOLEAN,
            processing_timestamp TIMESTAMP
        )
        STORED AS PARQUET
        """
        
        cursor.execute(create_table_query)
        logger.info(f"Ensured table {hive_output_table} exists")
        
        # Insert results
        records_saved = 0
        for _, row in df.iterrows():
            insert_query = f"""
            INSERT INTO {hive_output_table} VALUES (
                {row.get('id', 'NULL')},
                '{row.get('left', '').replace("'", "''")}',
                '{row.get('right', '').replace("'", "''")}',
                {row.get('match_probability', 0.0)},
                {'true' if row.get('match', False) else 'false'},
                current_timestamp()
            )
            """
            cursor.execute(insert_query)
            records_saved += 1
        
        cursor.close()
        connection.close()
        
        logger.info(f"Saved {records_saved} records to {hive_output_table}")
        return (records_saved,)
        
    except Exception as e:
        logger.error(f"Error saving results to Hive: {e}")
        raise

@dsl.pipeline(
    name='hive-ditto-matching-pipeline',
    description='Extract data from Hive, run DITTO entity matching, and save results back to Hive'
)
def hive_ditto_pipeline(
    hive_input_table: str = "base.table",
    hive_output_table: str = "default.ditto_matches",
    hive_host: str = "localhost",
    hive_port: int = 10000,
    hive_database: str = "default",
    ditto_task: str = "person_records",
    ditto_lm: str = "bert",
    ditto_max_len: int = 64,
    ditto_checkpoint_path: str = "checkpoints/"
):
    """
    Complete pipeline for Hive data extraction, DITTO matching, and result storage.
    
    Args:
        hive_input_table: Source Hive table
        hive_output_table: Destination Hive table for results
        hive_host: Hive server host
        hive_port: Hive server port
        hive_database: Hive database
        ditto_task: DITTO task configuration
        ditto_lm: Language model for DITTO
        ditto_max_len: Maximum sequence length
        ditto_checkpoint_path: Path to DITTO model checkpoints
    """
    
    # Create component operations
    extract_op = create_component_from_func(
        extract_hive_data_op,
        base_image="your-ditto-image:latest",  # Use your DITTO Docker image
        packages_to_install=['pyhive', 'pandas', 'PyHive[hive]']
    )
    
    matching_op = create_component_from_func(
        run_ditto_matching_op,
        base_image="your-ditto-image:latest"
    )
    
    save_op = create_component_from_func(
        save_results_to_hive_op,
        base_image="your-ditto-image:latest",
        packages_to_install=['pyhive', 'pandas', 'PyHive[hive]']
    )
    
    # Step 1: Extract data from Hive
    extract_task = extract_op(
        hive_table=hive_input_table,
        hive_host=hive_host,
        hive_port=hive_port,
        hive_database=hive_database
    )
    
    # Step 2: Run DITTO matching
    matching_task = matching_op(
        input_jsonl_path=extract_task.outputs['output_path'],
        task=ditto_task,
        lm=ditto_lm,
        max_len=ditto_max_len,
        checkpoint_path=ditto_checkpoint_path
    )
    
    # Step 3: Save results back to Hive
    save_task = save_op(
        input_jsonl_path=matching_task.outputs['output_path'],
        hive_output_table=hive_output_table,
        hive_host=hive_host,
        hive_port=hive_port,
        hive_database=hive_database
    )
    
    # Set dependencies
    matching_task.after(extract_task)
    save_task.after(matching_task)

if __name__ == '__main__':
    # Compile the pipeline
    kfp.compiler.Compiler().compile(hive_ditto_pipeline, 'hive_ditto_pipeline.yaml')
    print("Pipeline compiled to 'hive_ditto_pipeline.yaml'")
    
    # Example of how to run the pipeline
    # client = kfp.Client()
    # experiment = client.create_experiment('ditto-matching')
    # run = client.run_pipeline(
    #     experiment.id,
    #     'hive-ditto-matching',
    #     'hive_ditto_pipeline.yaml',
    #     arguments={
    #         'hive_input_table': 'base.table',
    #         'hive_output_table': 'results.person_matches'
    #     }
    # )