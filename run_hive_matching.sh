#!/bin/bash

# Complete workflow script for Hive data extraction, DITTO matching, and result storage
set -e

# Configuration
HIVE_HOST=${HIVE_HOST:-"localhost"}
HIVE_PORT=${HIVE_PORT:-10000}
HIVE_DATABASE=${HIVE_DATABASE:-"default"}
INPUT_TABLE=${INPUT_TABLE:-"base.table"}
OUTPUT_TABLE=${OUTPUT_TABLE:-"ditto_results.person_matches"}

# File paths
TEMP_DIR="./temp_matching"
INPUT_JSONL="$TEMP_DIR/input_data.jsonl"
OUTPUT_JSONL="$TEMP_DIR/output_results.jsonl"

# DITTO configuration
DITTO_TASK=${DITTO_TASK:-"person_records"}
DITTO_LM=${DITTO_LM:-"bert"}
DITTO_MAX_LEN=${DITTO_MAX_LEN:-64}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"checkpoints/"}

echo "=== Hive DITTO Matching Workflow ==="
echo "Input table: $INPUT_TABLE"
echo "Output table: $OUTPUT_TABLE"
echo "Hive server: $HIVE_HOST:$HIVE_PORT"
echo "Database: $HIVE_DATABASE"
echo ""

# Create temp directory
mkdir -p $TEMP_DIR

# Step 1: Extract data from Hive
echo "Step 1: Extracting data from Hive table: $INPUT_TABLE"
python hive_data_extractor.py \
    --mode extract \
    --hive_table "$INPUT_TABLE" \
    --file_path "$INPUT_JSONL" \
    --hive_host "$HIVE_HOST" \
    --hive_port "$HIVE_PORT" \
    --hive_database "$HIVE_DATABASE"

if [ ! -f "$INPUT_JSONL" ]; then
    echo "Error: Failed to extract data from Hive"
    exit 1
fi

echo "✓ Data extracted to: $INPUT_JSONL"
echo ""

# Step 2: Run DITTO matching
echo "Step 2: Running DITTO entity matching"
CUDA_VISIBLE_DEVICES=0 python matcher.py \
    --task "$DITTO_TASK" \
    --input_path "$INPUT_JSONL" \
    --output_path "$OUTPUT_JSONL" \
    --lm "$DITTO_LM" \
    --max_len "$DITTO_MAX_LEN" \
    --use_gpu \
    --fp16 \
    --checkpoint_path "$CHECKPOINT_PATH"

if [ ! -f "$OUTPUT_JSONL" ]; then
    echo "Error: DITTO matching failed"
    exit 1
fi

echo "✓ Matching completed. Results saved to: $OUTPUT_JSONL"
echo ""

# Step 3: Save results back to Hive
echo "Step 3: Saving results to Hive table: $OUTPUT_TABLE"
python hive_data_extractor.py \
    --mode save \
    --hive_table "$OUTPUT_TABLE" \
    --file_path "$OUTPUT_JSONL" \
    --hive_host "$HIVE_HOST" \
    --hive_port "$HIVE_PORT" \
    --hive_database "$HIVE_DATABASE"

echo "✓ Results saved to Hive table: $OUTPUT_TABLE"
echo ""

# Step 4: Show summary
echo "=== Workflow Summary ==="
if [ -f "$INPUT_JSONL" ]; then
    INPUT_COUNT=$(wc -l < "$INPUT_JSONL")
    echo "Records processed: $INPUT_COUNT"
fi

if [ -f "$OUTPUT_JSONL" ]; then
    OUTPUT_COUNT=$(wc -l < "$OUTPUT_JSONL")
    MATCHES=$(grep -c '"match":true' "$OUTPUT_JSONL" || echo "0")
    echo "Results generated: $OUTPUT_COUNT"
    echo "Matches found: $MATCHES"
fi

echo ""
echo "✓ Workflow completed successfully!"
echo "✓ Results are now available in Hive table: $OUTPUT_TABLE"

# Optional: Clean up temp files
# rm -rf $TEMP_DIR