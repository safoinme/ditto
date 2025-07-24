#!/bin/bash

# DITTO Matcher Test Script - No Hive Required
# This script tests just the matcher.py functionality with sample data

set -e

# Default values
DOCKER_IMAGE="172.17.232.16:9001/ditto-notebook:2.0"
CHECKPOINTS_PATH="./checkpoints"
OUTPUT_PATH="./output"
USE_GPU=true
NUM_SAMPLE_PAIRS=10

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

DITTO Matcher Test - Tests matcher.py without Hive dependencies

Options:
    --docker-image IMAGE         Docker image to use (default: $DOCKER_IMAGE)
    --checkpoints-path PATH      Local path to checkpoints directory (default: $CHECKPOINTS_PATH)
    --output-path PATH           Local path for output directory (default: $OUTPUT_PATH)
    --num-sample-pairs N         Number of sample pairs to test (default: $NUM_SAMPLE_PAIRS)
    --input-file FILE            Use custom input file instead of sample data
    --no-gpu                     Disable GPU acceleration
    --cpu-only                   Force CPU-only mode
    --model-task TASK            Model task (default: person_records)
    --output-file FILE           Save results to JSON file (relative to output path)
    --help                       Show this help message

Examples:
    # Basic test with 10 sample pairs
    $0

    # Test with more sample pairs
    $0 --num-sample-pairs 20

    # CPU-only test
    $0 --cpu-only --num-sample-pairs 5

    # Test with custom input file
    $0 --input-file test_data.jsonl --output-file results.json

    # Quick test to verify matcher.py works
    $0 --num-sample-pairs 3 --output-file quick_test.json

Test Data Format:
    If using --input-file, the file should be JSONL format with each line containing:
    ["left record text", "right record text"]
    
    Example:
    ["COL name VAL John Smith COL age VAL 30", "COL name VAL J. Smith COL age VAL 30"]
    ["COL name VAL Alice Johnson COL age VAL 25", "COL name VAL Bob Wilson COL age VAL 35"]

EOF
}

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --docker-image)
            DOCKER_IMAGE="$2"
            shift 2
            ;;
        --checkpoints-path)
            CHECKPOINTS_PATH="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --num-sample-pairs)
            NUM_SAMPLE_PAIRS="$2"
            EXTRA_ARGS+=(--num-sample-pairs "$2")
            shift 2
            ;;
        --input-file)
            EXTRA_ARGS+=(--input-file "/app/input/$2")
            INPUT_FILE="$2"
            shift 2
            ;;
        --no-gpu|--cpu-only)
            USE_GPU=false
            EXTRA_ARGS+=(--no-gpu)
            shift
            ;;
        --model-task)
            EXTRA_ARGS+=(--model-task "$2")
            shift 2
            ;;
        --output-file)
            EXTRA_ARGS+=(--output-file "/app/output/$2")
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Create necessary directories
mkdir -p "$OUTPUT_PATH"
if [[ ! -d "$CHECKPOINTS_PATH" ]]; then
    echo "Warning: Checkpoints directory $CHECKPOINTS_PATH does not exist"
    echo "Make sure you have the DITTO model checkpoints available"
fi

# Build Docker run command
DOCKER_CMD=(docker run --rm -it)

# GPU configuration
if [[ "$USE_GPU" == "true" ]]; then
    echo "=== GPU Mode Enabled ==="
    
    # Check if nvidia-docker runtime is available
    if docker info 2>/dev/null | grep -q nvidia; then
        DOCKER_CMD+=(--gpus all)
        echo "✓ NVIDIA Docker runtime detected"
    elif command -v nvidia-docker &> /dev/null; then
        # Fallback to nvidia-docker command
        DOCKER_CMD=(nvidia-docker run --rm -it)
        echo "✓ Using nvidia-docker command"
    else
        echo "⚠️  Warning: No NVIDIA Docker runtime detected"
        echo "   GPU acceleration may not work properly"
        echo "   Install nvidia-docker2 or Docker with NVIDIA Container Toolkit"
        DOCKER_CMD+=(--gpus all)  # Try anyway
    fi
else
    echo "=== CPU Mode ==="
fi

# Volume mounts
DOCKER_CMD+=(-v "$PWD/test_matcher_docker.py:/app/test_matcher_docker.py")
DOCKER_CMD+=(-v "$CHECKPOINTS_PATH:/checkpoints")
DOCKER_CMD+=(-v "$OUTPUT_PATH:/app/output")

# Mount input file if specified
if [[ -n "$INPUT_FILE" ]]; then
    INPUT_DIR=$(dirname "$INPUT_FILE")
    mkdir -p "$INPUT_DIR"
    DOCKER_CMD+=(-v "$PWD/$INPUT_DIR:/app/input")
fi

# Memory and CPU limits (smaller for testing)
DOCKER_CMD+=(--memory=8g)
DOCKER_CMD+=(--cpus=2)

# Environment variables
DOCKER_CMD+=(-e CUDA_VISIBLE_DEVICES=0)
DOCKER_CMD+=(-e CUDA_DEVICE_ORDER=PCI_BUS_ID)

# Working directory
DOCKER_CMD+=(-w /app)

# Docker image
DOCKER_CMD+=("$DOCKER_IMAGE")

# Command to run
DOCKER_CMD+=(python /app/test_matcher_docker.py "${EXTRA_ARGS[@]}")

# Show configuration
echo "=== Configuration ==="
echo "Docker Image: $DOCKER_IMAGE"
echo "Checkpoints Path: $CHECKPOINTS_PATH"
echo "Output Path: $OUTPUT_PATH"
echo "GPU Enabled: $USE_GPU"
echo "Sample Pairs: ${NUM_SAMPLE_PAIRS}"
echo "Input File: ${INPUT_FILE:-"Generated sample data"}"
echo ""

# Show the actual Docker command (for debugging)
echo "=== Docker Command ==="
echo "${DOCKER_CMD[*]}"
echo ""

# Run the Docker command
echo "=== Starting DITTO Matcher Test ==="
exec "${DOCKER_CMD[@]}" 