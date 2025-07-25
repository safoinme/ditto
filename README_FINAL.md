# DITTO Entity Matching Pipeline

A production-ready implementation of DITTO (Deep Learning for Entity Matching) integrated with Kubeflow Pipelines for scalable entity matching workflows.

## 🚀 Quick Start

### Generate Pipeline YAML
```bash
python3 ditto_kubeflow_pipeline.py --compile \
  --input-table preprocessed_analytics.model_reference \
  --hive-host 172.17.235.21 \
  --output ditto_kubeflow_pipeline.yaml
```

### Deploy to Kubeflow
1. Upload `dittofinalpipeline4.yaml` to Kubeflow Pipelines UI
2. Create a new run
3. Configure parameters in the UI (see Parameter Configuration section)

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Training DITTO Models](#training-ditto-models)
- [Pipeline Configuration](#pipeline-configuration)
- [Docker Images](#docker-images)
- [Parameter Configuration](#parameter-configuration)
- [Troubleshooting](#troubleshooting)
- [File Structure](#file-structure)

## 🏗️ Architecture Overview

### Pipeline Components

The DITTO Kubeflow pipeline consists of 4 main steps:

1. **Extract Data from Hive** - Queries Hive tables and converts to DITTO format
2. **Run DITTO Matching** - Executes entity matching using trained models
3. **Save Results to Hive** - Stores matching results back to Hive
4. **Create Log Summary** - Generates execution summary

### Data Flow

```
Hive Table → DITTO Format → GPU Matching → Results → Hive Storage
     ↓              ↓            ↓           ↓          ↓
[Extract]    [Convert]    [Match]     [Process]   [Store]
```

## 🧠 Training DITTO Models

### Using the Training Notebook

The primary way to train DITTO models is through the provided Jupyter notebook:

**`Hive_DITTO_Testing_Notebook.ipynb`**

This notebook contains:
- Complete training workflow
- Data preprocessing examples
- Model evaluation metrics
- Hyperparameter tuning guidance

### Training Steps

1. **Start Kubeflow Notebook Server**
   - Go to Kubeflow Central Dashboard → Notebooks
   - Create a new notebook server with:
     - **Image**: `172.17.232.16:9001/ditto-notebook:2.0`
     - **GPU**: Enable if available
     - **CPU**: 4+ cores recommended
     - **Memory**: 16Gi+ recommended
   - Connect to the notebook server

2. **Open Training Notebook**
   - Navigate to `Hive_DITTO_Testing_Notebook.ipynb`
   - Follow the step-by-step training process
   - Adjust hyperparameters as needed

3. **Key Training Parameters**
   ```python
   # Model Configuration
   model_name = "bert-base-uncased"
   max_length = 64
   learning_rate = 5e-5
   batch_size = 16
   epochs = 15
   
   # Data Configuration  
   task = "person_records"  # Your custom task name
   train_file = "data/train.txt"
   valid_file = "data/valid.txt"
   ```

4. **Save Trained Model**
   - Models are saved to `checkpoints/{task_name}/`
   - These checkpoints are embedded in the Docker image
   - Pipeline uses these checkpoints automatically

### Custom Dataset Training

For your own data:

1. **Data Format** - Convert to DITTO format:
   ```
   ["COL name VAL john smith", "COL name VAL j smith"]	1
   ["COL email VAL john@ex.com", "COL email VAL jane@ex.com"]	0
   ```

2. **Training Script**
   ```bash
   python train_ditto.py \
     --task your_task_name \
     --input_path data/your_train.txt \
     --max_len 64 \
     --lm bert \
     --use_gpu \
     --fp16
   ```

## ⚙️ Pipeline Configuration

### Main Pipeline File
**`ditto_kubeflow_pipeline.py`** - Production-ready pipeline

### Key Features
- **GPU Support** - Automatic GPU detection and fallback
- **Hive Integration** - Direct connection to Hive data warehouse
- **Volume Management** - Persistent storage for inter-step data
- **Error Handling** - Comprehensive error handling and logging
- **Configurable Parameters** - All parameters configurable via Kubeflow UI

### Pipeline Generation
```bash
python3 ditto_kubeflow_pipeline.py --compile \
  --input-table YOUR_TABLE_NAME \
  --hive-host YOUR_HIVE_HOST \
  --output YOUR_PIPELINE_NAME.yaml
```

### Optional Parameters
- `--no-cache` - Disable step caching
- `--input-table` - Specify input Hive table
- `--hive-host` - Hive server hostname
- `--output` - Output YAML filename

## 🐳 Docker Images

### Production Image
**`172.17.232.16:9001/ditto-notebook:2.0`**

Built from `Dockerfile.kubeflow`:
- CUDA 12.2 support
- PyTorch with GPU acceleration
- Pre-trained BERT models
- Hive connectivity (pyhive)
- Kubeflow Pipeline SDK
- Embedded checkpoints at `/home/jovyan/checkpoints/`

### Build Commands
```bash
# Build the image
docker build -f Dockerfile.kubeflow -t 172.17.232.16:9001/ditto-notebook:2.0 .

# Push to registry
docker push 172.17.232.16:9001/ditto-notebook:2.0
```

### Image Contents
```
/home/jovyan/
├── checkpoints/           # Pre-trained model checkpoints
│   └── person_records/    # Your trained models
├── models/               # BERT base models
│   ├── bert-base-uncased/
│   └── distilbert-base-uncased/
├── matcher.py            # Core matching script
└── [other project files]
```

## 📊 Parameter Configuration

When creating a run in Kubeflow UI, configure these parameters:

### Hive Connection
```yaml
hive_host: "172.17.235.21"
hive_port: 10000
hive_user: "lhimer"
hive_database: "preprocessed_analytics"
```

### Data Parameters
```yaml
input_table: "preprocessed_analytics.model_reference"
sample_limit: null  # or integer for testing
matching_mode: "auto"  # or "production"/"testing"
```

### DITTO Model Parameters
```yaml
model_task: "person_records"
checkpoint_path: "/home/jovyan/checkpoints"
lm: "bert"
max_len: 64
use_gpu: true
fp16: true  # Mixed precision for faster GPU training
summarize: false
```

### Output Parameters
```yaml
save_to_hive: false  # Set to true to save results
output_table: "ditto_matching_results"
```

### Performance Tuning
- **GPU Memory**: Set `use_gpu: false` if GPU memory issues
- **Batch Processing**: Use `sample_limit` for testing with small datasets
- **Mixed Precision**: Keep `fp16: true` for faster GPU processing

## 🔧 Troubleshooting

### Common Issues

**1. GPU Out of Memory**
```yaml
# Solution: Disable GPU or reduce batch size
use_gpu: false
```

**2. Hive Connection Timeout**
```yaml
# Solution: Verify Hive host and credentials
hive_host: "YOUR_CORRECT_HOST"
hive_user: "YOUR_USERNAME"
```

**3. Missing Checkpoints**
```bash
# Solution: Rebuild Docker image with checkpoints
docker build -f Dockerfile.kubeflow -t YOUR_IMAGE:TAG .
```

**4. Volume Binding Issues**
- Use `ReadWriteOnce` instead of `ReadWriteMany`
- Ensure sufficient storage quota

### Debug Mode
Enable detailed logging by checking pipeline logs in Kubeflow UI.

### Resource Requirements
- **CPU**: 4 cores minimum
- **Memory**: 16Gi minimum
- **GPU**: Optional but recommended (NVIDIA with CUDA 12.2)
- **Storage**: 10Gi for pipeline data

## 📁 File Structure

```
ditto/
├── README_FINAL.md                    # This file
├── ditto_kubeflow_pipeline.py         # 🔥 Main pipeline (PRODUCTION)
├── Hive_DITTO_Testing_Notebook.ipynb  # 📚 Training notebook
├── Dockerfile.kubeflow                # 🐳 Production Docker image
├── matcher.py                         # Core matching logic
├── checkpoints/                       # Pre-trained models
│   └── person_records/
├── models/                            # BERT models
│   └── bert-base-uncased/
└── data/                              # Training datasets
    └── er_magellan/
```

### Key Files

- **`ditto_kubeflow_pipeline.py`** - Main production pipeline
- **`Hive_DITTO_Testing_Notebook.ipynb`** - Training and experimentation
- **`Dockerfile.kubeflow`** - Production Docker image definition
- **`matcher.py`** - Core DITTO matching logic
- **`requirements.txt`** - Python dependencies

### Deprecated Files
- `ditto_kubeflow_pipeline_gpu_enabled.py` - Superseded by main pipeline
- `ditto_kubeflow_pipeline_no_pvc.py` - Old version without volumes

## 🎯 Best Practices

### Training
1. Start with the provided notebook for learning
2. Use small datasets for initial experimentation  
3. Scale up gradually with your production data
4. Save checkpoints regularly during training

### Production Deployment
1. Test pipeline with `sample_limit` first
2. Monitor GPU memory usage
3. Use appropriate persistent volume sizes
4. Enable result saving only when needed

### Performance Optimization
1. Use GPU when available (`use_gpu: true`)
2. Enable mixed precision (`fp16: true`) 
3. Batch process large datasets
4. Cache pipeline steps when appropriate

## 📄 License

This project extends the original DITTO implementation for production use with Kubeflow Pipelines.

---

**Questions?** Check the training notebook or review the troubleshooting section above.