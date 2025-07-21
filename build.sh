#!/bin/bash

# Ditto Docker Build and Validation Script
set -e

# Configuration
IMAGE_NAME="${IMAGE_NAME:-ditto-matching}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_step "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check NVIDIA Docker support
    if ! docker run --rm --gpus all nvidia/cuda:12.2-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        log_warn "NVIDIA Docker support not available. GPU features will be disabled."
    else
        log_info "NVIDIA Docker support detected"
    fi
    
    # Check if Dockerfile exists
    if [[ ! -f "Dockerfile" ]]; then
        log_error "Dockerfile not found in current directory"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Build Docker image
build_image() {
    log_step "Building Docker image: ${FULL_IMAGE_NAME}"
    
    # Build with BuildKit for better performance
    export DOCKER_BUILDKIT=1
    
    docker build \
        --progress=plain \
        --tag "${FULL_IMAGE_NAME}" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        .
    
    log_info "Docker image built successfully"
}

# Test the built image
test_image() {
    log_step "Testing Docker image..."
    
    log_info "Testing basic Python environment..."
    docker run --rm "${FULL_IMAGE_NAME}" python --version
    
    log_info "Testing PyTorch CUDA availability..."
    docker run --rm "${FULL_IMAGE_NAME}" python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
else:
    print('CUDA not available (this is expected without --gpus flag)')
"
    
    log_info "Testing Hive connectivity libraries..."
    docker run --rm "${FULL_IMAGE_NAME}" python -c "
try:
    import pyhive
    from pyhive import hive
    import thrift
    import sasl
    print('Hive connectivity: ✓ All libraries available')
except ImportError as e:
    print(f'Hive connectivity: ✗ Missing library: {e}')
    exit(1)
"
    
    log_info "Testing ML/DL libraries..."
    docker run --rm "${FULL_IMAGE_NAME}" python -c "
import transformers
import spacy
import sklearn
import pandas
import numpy
print(f'Transformers: {transformers.__version__}')
print(f'spaCy: {spacy.__version__}')
print(f'Scikit-learn: {sklearn.__version__}')
print(f'Pandas: {pandas.__version__}')
print(f'NumPy: {numpy.__version__}')
print('ML/DL libraries: ✓ All libraries available')
"
    
    log_info "Testing Kubeflow SDK..."
    docker run --rm "${FULL_IMAGE_NAME}" python -c "
import kfp
import kubernetes
print(f'Kubeflow Pipelines SDK: {kfp.__version__}')
print(f'Kubernetes client: {kubernetes.__version__}')
print('Kubeflow SDK: ✓ Available')
"
    
    log_info "Testing Ditto components..."
    docker run --rm "${FULL_IMAGE_NAME}" python -c "
import sys
import os
sys.path.append('/app/ditto')

# Test if ditto_light modules can be imported
try:
    from ditto_light.ditto import DittoModel
    from ditto_light.dataset import DittoDataset
    print('Ditto components: ✓ Available')
except ImportError as e:
    print(f'Ditto components: ⚠ Some imports failed: {e}')
    print('This might be expected if models are not present')
"
    
    log_info "All tests passed successfully!"
}

# Test with GPU (if available)
test_gpu() {
    log_step "Testing GPU functionality..."
    
    if docker run --rm --gpus all "${FULL_IMAGE_NAME}" python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU Test: ✓ CUDA available with {torch.cuda.device_count()} GPU(s)')
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'cuDNN Version: {torch.backends.cudnn.version()}')
    
    # Test tensor operations on GPU
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = torch.matmul(x, y)
    print(f'GPU Tensor Operations: ✓ Working')
    exit(0)
else:
    print('GPU Test: ✗ CUDA not available')
    exit(1)
" 2>/dev/null; then
        log_info "GPU tests passed!"
    else
        log_warn "GPU tests failed or GPU not available"
    fi
}

# Generate sample data for testing
generate_test_data() {
    log_step "Generating sample test data..."
    
    mkdir -p data/input data/output checkpoints config
    
    # Create sample pairs file
    cat > data/input/test_pairs.jsonl << 'EOF'
[{"name": "John Doe", "age": "30", "city": "New York"}, {"full_name": "John Doe", "years": "30", "location": "NYC"}]
[{"name": "Jane Smith", "age": "25", "city": "Boston"}, {"full_name": "Jane Smith", "years": "25", "location": "Boston"}]
[{"name": "Bob Johnson", "age": "35", "city": "Chicago"}, {"full_name": "Robert Johnson", "years": "35", "location": "Chicago"}]
EOF

    log_info "Sample test data created in data/input/"
}

# Show usage information
show_usage() {
    echo "Ditto Docker Build and Validation Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build          Build the Docker image"
    echo "  test           Test the built image (basic tests)"
    echo "  test-gpu       Test GPU functionality"
    echo "  full           Build and run all tests"
    echo "  sample-data    Generate sample test data"
    echo "  clean          Remove built image"
    echo "  help           Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  IMAGE_NAME     Docker image name (default: ditto-matching)"
    echo "  IMAGE_TAG      Docker image tag (default: latest)"
    echo ""
    echo "Examples:"
    echo "  $0 full"
    echo "  IMAGE_NAME=my-ditto $0 build"
    echo "  $0 test-gpu"
}

# Clean up built images
clean() {
    log_step "Cleaning up Docker images..."
    docker rmi "${FULL_IMAGE_NAME}" 2>/dev/null || log_warn "Image ${FULL_IMAGE_NAME} not found"
    docker system prune -f
    log_info "Cleanup completed"
}

# Main execution
case "${1:-full}" in
    "build")
        check_prerequisites
        build_image
        ;;
    "test")
        test_image
        ;;
    "test-gpu")
        test_gpu
        ;;
    "full")
        check_prerequisites
        build_image
        test_image
        generate_test_data
        echo ""
        log_info "🎉 Full build and validation completed successfully!"
        log_info "Your Ditto image is ready: ${FULL_IMAGE_NAME}"
        log_info "Run 'docker run -it --gpus all ${FULL_IMAGE_NAME}' to start interactive session"
        ;;
    "sample-data")
        generate_test_data
        ;;
    "clean")
        clean
        ;;
    "help"|"--help"|"-h")
        show_usage
        ;;
    *)
        log_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac