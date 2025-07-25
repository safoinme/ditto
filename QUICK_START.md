# DITTO Pipeline Quick Start

## 🚀 Generate Pipeline YAML (Working Command)

```bash
python3 ditto_kubeflow_pipeline.py --compile \
  --input-table preprocessed_analytics.model_reference \
  --hive-host 172.17.235.21 \
  --output dittofinalpipeline4.yaml
```

## 📊 Kubeflow UI Parameters

When starting a run in Kubeflow UI, set these parameters:

### Required Parameters
- **hive_host**: `172.17.235.21`
- **input_table**: `preprocessed_analytics.model_reference` 
- **use_gpu**: `true` (or `false` if no GPU available)

### Optional Parameters  
- **sample_limit**: Leave empty for full dataset, or set integer for testing
- **model_task**: `person_records` (your trained model name)
- **checkpoint_path**: `/home/jovyan/checkpoints` (default)
- **save_to_hive**: `false` (set to `true` to save results)
- **output_table**: `ditto_matching_results` (if saving to Hive)

## 🐳 Docker Image
Current production image: `172.17.232.16:9001/ditto-notebook:2.0`

## 📚 Training
1. Create Kubeflow Notebook Server with image: `172.17.232.16:9001/ditto-notebook:2.0`
2. Open `Hive_DITTO_Testing_Notebook.ipynb` in the notebook server
3. Follow the training steps in the notebook

## ⚠️ Troubleshooting
- GPU issues → Set `use_gpu: false`
- Volume errors → Pipeline now uses `ReadWriteOnce`
- Missing matcher.py → Checkpoints are at `/home/jovyan/checkpoints/`