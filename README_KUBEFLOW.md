# Ditto Entity Matching with Kubeflow Pipelines

This directory contains the enhanced Ditto project with Kubeflow pipeline integration for automated entity matching using Hive data sources.

## Overview

The solution provides:
1. **Hive Data Extraction** - Downloads data from two Hive tables
2. **Data Pairing** - Creates cartesian product pairs for matching
3. **Ditto Matching** - Runs the pre-trained Ditto model for entity matching
4. **Result Storage** - Optionally saves results back to Hive

## Files Added/Modified

### New Components
- `hive_data_extractor.py` - Extracts data from Hive and creates pairs
- `ditto_kubeflow_pipeline.py` - Complete Kubeflow pipeline implementation
- `Dockerfile.kubeflow` - Enhanced Docker image with Hive and Kubeflow support
- `ditto-config.yaml` - Kubernetes configuration for deployment
- `README_KUBEFLOW.md` - This documentation

### Pipeline Architecture

```
[Hive Table 1] ──┐
                 ├─► [Data Extraction] ─► [Cartesian Pairing] ─► [Ditto Matching] ─► [Optional Hive Save]
[Hive Table 2] ──┘
```

## Quick Start

### 1. Build the Docker Image

```bash
cd /path/to/ditto
docker build -f Dockerfile.kubeflow -t your-registry/ditto-kubeflow:latest .
docker push your-registry/ditto-kubeflow:latest
```

### 2. Update Pipeline Configuration

Edit `ditto_kubeflow_pipeline.py` and replace `'your-registry/ditto-kubeflow:latest'` with your actual image registry.

### 3. Deploy Kubernetes Resources

```bash
kubectl apply -f ditto-config.yaml
```

### 4. Compile the Pipeline

```bash
python ditto_kubeflow_pipeline.py \
    --input-table "database.source_table" \
    --hive-host "your-hive-host" \
    --output "my-ditto-pipeline.yaml"
```

### 5. Upload and Run

Upload `my-ditto-pipeline.yaml` to your Kubeflow Pipelines UI and create a run with your parameters.

## Pipeline Parameters

### Required Parameters
- `input_table` - Input Hive table (format: database.table)
- `hive_host` - Hive server hostname/IP
- `hive_user` - Hive username

### Optional Parameters
- `hive_port` - Hive server port (default: 10000)
- `hive_database` - Hive database (default: "default")
- `sample_limit` - Limit rows from table (for testing)
- `model_task` - Ditto model task (default: "wdc_all_small")
- `lm` - Language model (default: "distilbert")
- `max_len` - Max sequence length (default: 64)
- `use_gpu` - Use GPU acceleration (default: true)
- `fp16` - Use half-precision (default: true)
- `dk` - Domain knowledge injection ("product" or "general")
- `summarize` - Use summarization optimization
- `save_to_hive` - Save results back to Hive (default: false)
- `output_table` - Output Hive table name

## Example Usage

### Basic Self-Matching
```bash
python ditto_kubeflow_pipeline.py \
    --input-table "sales_db.customers" \
    --hive-host "hive.company.com"
```

### With Optimizations
```bash
python ditto_kubeflow_pipeline.py \
    --input-table "sales_db.products" \
    --hive-host "hive.company.com" \
    --output "product-matching-pipeline.yaml"
```

## Pipeline Steps

### 1. Data Extraction (`extract_hive_data_func`)
- Connects to Hive using provided credentials
- Extracts data from specified table
- Removes table name prefixes from column names (tablename.column → column)
- Converts to DITTO COL/VAL format and creates self-matching pairs
- Saves to `/data/input/test_pairs.jsonl` in correct format for matcher.py

### 2. Ditto Matching (`run_ditto_matching_func`)
- Loads the specified pre-trained Ditto model
- Processes all pairs using the matcher.py script
- Applies domain knowledge and optimizations if specified
- Outputs results to `/data/output/matching_results.jsonl`

### 3. Result Storage (`save_results_to_hive_func`)
- Optionally saves matching results back to Hive
- Creates table structure with match scores and confidence
- Includes processing timestamps for audit trail

## Model Requirements

### Checkpoints
Place your trained Ditto model checkpoints in the `/checkpoints/` directory of your container or mount them as a volume.

### Supported Models
- `distilbert` (default)
- `bert`
- `albert`

## Resource Requirements

### Minimum
- CPU: 2 cores
- Memory: 4GB
- Storage: 10GB

### Recommended (with GPU)
- CPU: 4 cores
- Memory: 8GB
- GPU: 1x NVIDIA GPU (CUDA compatible)
- Storage: 20GB

## Troubleshooting

### Common Issues

1. **Hive Connection Failed**
   - Verify Hive host, port, and credentials
   - Check network connectivity from Kubernetes cluster
   - Ensure Hive service is running

2. **Model Not Found**
   - Check if model checkpoints are available in `/checkpoints/`
   - Verify the `model_task` parameter matches your trained model
   - Ensure model files are properly mounted

3. **Out of Memory**
   - Reduce `table1_limit` and `table2_limit` for testing
   - Decrease `max_len` parameter
   - Increase memory allocation for the matching step

4. **GPU Not Available**
   - Set `use_gpu=false` if no GPU is available
   - Verify GPU drivers and CUDA installation in container
   - Check Kubernetes GPU resource allocation

### Monitoring

The pipeline includes comprehensive logging and metrics:
- Processing counts and timing information
- Matching statistics (matches vs non-matches)
- Error messages and stack traces
- Resource utilization metrics

## Integration with Entity Project 5

This Ditto pipeline is designed to complement the existing Entity Project 5. Key differences:

- **Data Source**: Uses direct Hive extraction vs. Spark-based preprocessing
- **Matching Algorithm**: Uses pre-trained transformers vs. Splink statistical matching
- **Output Format**: JSONL with confidence scores vs. CSV with cluster IDs
- **Resource Usage**: GPU-optimized for deep learning vs. CPU-optimized for statistical processing

## Next Steps

1. Train domain-specific Ditto models for your data
2. Implement blocking strategies to reduce pair generation
3. Add data quality checks and validation
4. Create automated retraining pipelines
5. Integrate with MLflow for model versioning

## Support

For issues specific to this Kubeflow integration, check:
1. Container logs in Kubernetes
2. Kubeflow pipeline execution logs
3. Hive server logs for connection issues
4. GPU availability and CUDA compatibility