# Person Record Entity Matching Pipeline

This pipeline uses Ditto (Deep Entity Matching with Pre-Trained Language Models) to perform entity matching on person records. It includes data augmentation, format conversion, model training, and matching capabilities.

## Overview

The pipeline consists of 4 main steps:
1. **Data Augmentation**: Creates variations of your CSV data by modifying 1-2 fields per row
2. **Format Conversion**: Converts CSV data to Ditto's entity matching format
3. **Model Training**: Trains a transformer-based model for entity matching
4. **Entity Matching**: Runs predictions on new data pairs

## Files Created

- `augment_csv_data.py` - Data augmentation script
- `csv_to_ditto.py` - CSV to Ditto format converter
- `run_person_matching.py` - Complete pipeline orchestrator
- Updated `configs.json` with person_records configuration

## Prerequisites

Make sure you have the required dependencies installed:

```bash
# Install Python dependencies
pip install pandas scikit-learn

# Install Ditto requirements (if not already done)
pip install -r requirements.txt
conda install -c conda-forge nvidia-apex  # For fp16 training
python -m spacy download en_core_web_sm
```

## Quick Start

### Option 1: Run Complete Pipeline

Run everything in one command:

```bash
python run_person_matching.py --csv data.csv --complete
```

This will:
1. Augment your CSV data (10 variations per row)
2. Convert to Ditto format and create train/val/test splits
3. Train the entity matching model
4. Run sample predictions

### Option 2: Step-by-Step Execution

#### Step 1: Data Augmentation

```bash
python augment_csv_data.py data.csv --num_augmentations 10
```

This creates `data_augmented.csv` with 10 augmented examples for each original row.

#### Step 2: Convert to Ditto Format

```bash
python csv_to_ditto.py data_augmented.csv --output_dir data/person_records
```

This creates:
- `data/person_records/train.txt` (70% of data)
- `data/person_records/valid.txt` (15% of data)  
- `data/person_records/test.txt` (15% of data)

#### Step 3: Train the Model

```bash
python train_ditto.py \
  --task person_records \
  --batch_size 16 \
  --max_len 128 \
  --lr 3e-5 \
  --n_epochs 20 \
  --lm distilbert \
  --fp16 \
  --da del \
  --dk general \
  --summarize \
  --save_model \
  --logdir checkpoints
```

#### Step 4: Run Entity Matching

```bash
python matcher.py \
  --task person_records \
  --input_path input/sample_pairs.jsonl \
  --output_path output/predictions.jsonl \
  --lm distilbert \
  --max_len 128 \
  --use_gpu \
  --fp16 \
  --checkpoint_path checkpoints/ \
  --dk general \
  --summarize
```

## Individual Script Usage

### Data Augmentation Script

```bash
# Basic usage
python augment_csv_data.py data.csv

# Specify output file and number of augmentations
python augment_csv_data.py data.csv --output augmented_data.csv --num_augmentations 5

# Help
python augment_csv_data.py --help
```

**What it does:**
- Randomly modifies 1-2 fields per row with character-level changes:
  - Add random letters
  - Remove 1-2 characters  
  - Change 1-2 characters
- Creates 10 variations by default for each original row
- Preserves original data alongside augmented versions

### CSV to Ditto Converter

```bash
# Basic usage
python csv_to_ditto.py data.csv

# Specify output directory and match ratio
python csv_to_ditto.py data.csv --output_dir my_data --match_ratio 0.2

# Help
python csv_to_ditto.py --help
```

**What it does:**
- Converts CSV records to Ditto's serialized format
- Creates positive pairs (likely matches) and negative pairs (non-matches)
- Uses similarity heuristics based on names and IDs
- Splits data into train (70%), validation (15%), and test (15%) sets

### Pipeline Orchestrator

```bash
# Complete pipeline
python run_person_matching.py --csv data.csv --complete

# Data preparation only
python run_person_matching.py --csv data.csv --prepare-only

# Training only (if data already prepared)
python run_person_matching.py --train-only

# Matching only
python run_person_matching.py --match-only --input input/my_pairs.jsonl

# Help
python run_person_matching.py --help
```

## Input Data Format

Your CSV should have the following columns (as in your `data.csv`):
- `ifu_clean` - IFU identifier
- `num_cin_clean` - CIN number
- `num_ce_clean` - CE number  
- `num_ppr_clean` - PPR number
- `num_cnss_clean` - CNSS number
- `nom_clean` - Last name
- `nom_prenom_rs_clean` - Full name

## Output Formats

### Ditto Format Files
Each line contains: `<entry1>\t<entry2>\t<label>`

Example:
```
COL ifu_id VAL 51684570 COL ce_number VAL A071215G COL lastname VAL COULIBALY COL fullname VAL COULIBALY ZIE SOULEYMANE AE	COL ifu_id VAL 52684570 COL ce_number VAL A071216G COL lastname VAL COULIBALY COL fullname VAL COULIBALY ZIE SOULEYMANE	1
```

### Prediction Files (JSONL)
Input format for matching:
```json
{"id": "pair_1", "left": "COL lastname VAL SMITH ...", "right": "COL lastname VAL SMYTH ..."}
```

Output format:
```json
{"id": "pair_1", "match_confidence": 0.85, "match": true}
```

## Model Configuration

The model uses these optimizations:
- **Language Model**: DistilBERT (faster than BERT, good performance)
- **Data Augmentation**: 'del' operator (removes token spans)
- **Domain Knowledge**: 'general' injection (NER tagging, number normalization)
- **Summarization**: Retains high TF-IDF tokens within sequence length
- **FP16**: Half-precision training for faster GPU training

## Customization

### Adjust Matching Logic
Edit the `are_likely_matches()` method in `csv_to_ditto.py` to change how positive pairs are created.

### Change Augmentation Operations
Modify the `augment_field()` method in `augment_csv_data.py` to add different character-level modifications.

### Model Hyperparameters
Adjust training parameters in `run_person_matching.py` or use direct `train_ditto.py` calls with different parameters.

## Troubleshooting

### Memory Issues
- Reduce `--batch_size` from 16 to 8 or 4
- Reduce `--max_len` from 128 to 64
- Disable `--fp16` if using CPU

### Poor Performance
- Increase `--n_epochs` (try 30-40)
- Try different language models: `--lm bert` or `--lm roberta`
- Adjust `--match_ratio` in conversion (try 0.2 for more negatives)

### CUDA Issues
- Remove `--fp16` and `--use_gpu` flags to run on CPU
- Check CUDA installation with `python -c "import torch; print(torch.cuda.is_available())"`

## Expected Results

With your dataset of ~108 person records:
- **Augmented dataset**: ~1,188 records (108 original + 1,080 augmented)
- **Entity pairs**: ~2,000-5,000 pairs (depends on similarity patterns)
- **Training time**: 5-15 minutes on GPU, 30-60 minutes on CPU
- **Expected F1 score**: 0.80-0.95 (depending on data quality and similarity patterns)

## File Structure After Running

```
ditto/
├── data/
│   └── person_records/
│       ├── train.txt
│       ├── valid.txt
│       └── test.txt
├── checkpoints/
│   └── person_records/
│       └── model.pt
├── input/
│   └── sample_pairs.jsonl
├── output/
│   └── predictions.jsonl
├── data_augmented.csv
├── augment_csv_data.py
├── csv_to_ditto.py
├── run_person_matching.py
└── PERSON_MATCHING_README.md
```

## Next Steps

1. Run the complete pipeline on your data
2. Evaluate results on the test set
3. Adjust hyperparameters if needed
4. Create custom input files for your specific matching needs
5. Integrate the trained model into your production system 