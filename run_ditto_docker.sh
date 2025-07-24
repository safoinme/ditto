#!/bin/bash

# DITTO Entity Matching Pipeline - Docker Runner Script
# This script runs the DITTO pipeline using Docker instead of Kubernetes

set -e

# Default values
DOCKER_IMAGE="172.17.232.16:9001/ditto-notebook:2.0"
CHECKPOINTS_PATH="./checkpoints"
OUTPUT_PATH="./output"
USE_GPU=true
SAMPLE_LIMIT=""
MATCHING_MODE="auto"

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

DITTO Entity Matching Pipeline - Docker Runner

Options:
    --docker-image IMAGE         Docker image to use (default: $DOCKER_IMAGE)
    --checkpoints-path PATH      Local path to checkpoints directory (default: $CHECKPOINTS_PATH)
    --output-path PATH           Local path for output directory (default: $OUTPUT_PATH)
    --sample-limit N             Limit number of records to process
    --matching-mode MODE         Matching mode: auto, production, testing (default: $MATCHING_MODE)
    --no-gpu                     Disable GPU acceleration
    --cpu-only                   Force CPU-only mode
    --hive-host HOST             Hive server host (default: from script)
    --hive-port PORT             Hive server port (default: from script)
    --hive-user USER             Hive username (default: from script)
    --input-table TABLE          Input table name (default: from script)
    --output-file FILE           Save results to JSON file (relative to output path)
    --save-to-hive               Save results back to Hive
    --help                       Show this help message

Examples:
    # Basic run with GPU (default)
    $0

    # CPU-only run with sample limit
    $0 --cpu-only --sample-limit 1000

    # Save results to file
    $0 --output-file results.json --sample-limit 100

    # Production mode with custom checkpoints
    $0 --matching-mode production --checkpoints-path /path/to/my/checkpoints

GPU Requirements:
    For GPU support, you need:
    - NVIDIA Docker runtime installed
    - Compatible NVIDIA drivers
    - CUDA-compatible GPU

    If GPU is not available, the script will automatically fall back to CPU mode.

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
        --sample-limit)
            SAMPLE_LIMIT="$2"
            EXTRA_ARGS+=(--sample-limit "$2")
            shift 2
            ;;
        --matching-mode)
            MATCHING_MODE="$2"
            EXTRA_ARGS+=(--matching-mode "$2")
            shift 2
            ;;
        --no-gpu|--cpu-only)
            USE_GPU=false
            EXTRA_ARGS+=(--no-gpu)
            shift
            ;;
        --hive-host)
            EXTRA_ARGS+=(--hive-host "$2")
            shift 2
            ;;
        --hive-port)
            EXTRA_ARGS+=(--hive-port "$2")
            shift 2
            ;;
        --hive-user)
            EXTRA_ARGS+=(--hive-user "$2")
            shift 2
            ;;
        --input-table)
            EXTRA_ARGS+=(--input-table "$2")
            shift 2
            ;;
        --output-file)
            EXTRA_ARGS+=(--output-file "/app/output/$2")
            shift 2
            ;;
        --save-to-hive)
            EXTRA_ARGS+=(--save-to-hive)
            shift
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
DOCKER_CMD+=(-v "$PWD/docker_ditto_runner.py:/app/docker_ditto_runner.py")
DOCKER_CMD+=(-v "$CHECKPOINTS_PATH:/checkpoints")
DOCKER_CMD+=(-v "$OUTPUT_PATH:/app/output")

# Memory and CPU limits
DOCKER_CMD+=(--memory=16g)
DOCKER_CMD+=(--cpus=4)

# Environment variables
DOCKER_CMD+=(-e CUDA_VISIBLE_DEVICES=0)
DOCKER_CMD+=(-e CUDA_DEVICE_ORDER=PCI_BUS_ID)

# Working directory
DOCKER_CMD+=(-w /app)

# Docker image
DOCKER_CMD+=("$DOCKER_IMAGE")

# Command to run
DOCKER_CMD+=(python /app/docker_ditto_runner.py "${EXTRA_ARGS[@]}")

# Show configuration
echo "=== Configuration ==="
echo "Docker Image: $DOCKER_IMAGE"
echo "Checkpoints Path: $CHECKPOINTS_PATH"
echo "Output Path: $OUTPUT_PATH"
echo "GPU Enabled: $USE_GPU"
echo "Sample Limit: ${SAMPLE_LIMIT:-"No limit"}"
echo "Matching Mode: $MATCHING_MODE"
echo ""

# Show the actual Docker command (for debugging)
echo "=== Docker Command ==="
echo "${DOCKER_CMD[*]}"
echo ""

# Confirm before running
read -p "Continue with this configuration? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Run the Docker command
echo "=== Starting DITTO Pipeline ==="
exec "${DOCKER_CMD[@]}" 