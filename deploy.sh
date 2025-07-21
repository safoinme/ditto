#!/bin/bash

# Ditto Kubeflow Pipeline Deployment Script
set -e

# Configuration
REGISTRY="${DOCKER_REGISTRY:-your-registry}"
IMAGE_NAME="ditto-kubeflow"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

# Default values
HIVE_HOST="${HIVE_HOST:-localhost}"
HIVE_PORT="${HIVE_PORT:-10000}"
HIVE_USER="${HIVE_USER:-hive}"
HIVE_DATABASE="${HIVE_DATABASE:-default}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if docker is available
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if Python is available
    if ! command -v python &> /dev/null; then
        log_error "Python is not installed or not in PATH"
        exit 1
    fi
    
    # Check if required Python packages are available
    python -c "import kfp" 2>/dev/null || {
        log_warn "Kubeflow Pipelines SDK not found. Installing..."
        pip install kfp
    }
    
    log_info "Prerequisites check completed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image: ${FULL_IMAGE_NAME}"
    
    # Check if Dockerfile.kubeflow exists
    if [[ ! -f "Dockerfile.kubeflow" ]]; then
        log_error "Dockerfile.kubeflow not found in current directory"
        exit 1
    fi
    
    # Build the image
    docker build -f Dockerfile.kubeflow -t "${FULL_IMAGE_NAME}" .
    
    log_info "Docker image built successfully"
}

# Push Docker image
push_image() {
    log_info "Pushing Docker image to registry..."
    docker push "${FULL_IMAGE_NAME}"
    log_info "Docker image pushed successfully"
}

# Update pipeline configuration
update_pipeline_config() {
    log_info "Updating pipeline configuration..."
    
    # Replace placeholder in pipeline file
    if [[ -f "ditto_kubeflow_pipeline.py" ]]; then
        sed -i.bak "s|'your-registry/ditto-kubeflow:latest'|'${FULL_IMAGE_NAME}'|g" ditto_kubeflow_pipeline.py
        log_info "Pipeline configuration updated"
    else
        log_error "ditto_kubeflow_pipeline.py not found"
        exit 1
    fi
}

# Deploy Kubernetes resources
deploy_k8s_resources() {
    log_info "Deploying Kubernetes resources..."
    
    if [[ -f "ditto-config.yaml" ]]; then
        kubectl apply -f ditto-config.yaml
        log_info "Kubernetes resources deployed"
    else
        log_warn "ditto-config.yaml not found, skipping Kubernetes resource deployment"
    fi
}

# Compile pipeline
compile_pipeline() {
    local table1="$1"
    local table2="$2"
    local output_file="${3:-ditto-matching-pipeline.yaml}"
    
    log_info "Compiling Kubeflow pipeline..."
    
    if [[ -z "$table1" || -z "$table2" ]]; then
        log_error "Table names are required for pipeline compilation"
        echo "Usage: $0 compile <table1> <table2> [output_file]"
        exit 1
    fi
    
    python ditto_kubeflow_pipeline.py \
        --table1 "$table1" \
        --table2 "$table2" \
        --hive-host "$HIVE_HOST" \
        --output "$output_file"
    
    log_info "Pipeline compiled to: $output_file"
}

# Show usage
show_usage() {
    echo "Ditto Kubeflow Pipeline Deployment Script"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  build              Build Docker image"
    echo "  push               Push Docker image to registry"
    echo "  deploy             Deploy Kubernetes resources"
    echo "  compile <t1> <t2>  Compile pipeline for tables t1 and t2"
    echo "  full <t1> <t2>     Full deployment (build, push, deploy, compile)"
    echo "  help               Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  DOCKER_REGISTRY    Docker registry URL (default: your-registry)"
    echo "  IMAGE_TAG          Docker image tag (default: latest)"
    echo "  HIVE_HOST          Hive server host (default: localhost)"
    echo "  HIVE_PORT          Hive server port (default: 10000)"
    echo "  HIVE_USER          Hive username (default: hive)"
    echo "  HIVE_DATABASE      Hive database (default: default)"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 full sales_db.customers crm_db.contacts"
    echo "  DOCKER_REGISTRY=my-registry.com $0 build"
}

# Main execution
case "${1:-}" in
    "build")
        check_prerequisites
        build_image
        ;;
    "push")
        check_prerequisites
        push_image
        ;;
    "deploy")
        check_prerequisites
        deploy_k8s_resources
        ;;
    "compile")
        check_prerequisites
        compile_pipeline "$2" "$3" "$4"
        ;;
    "full")
        if [[ -z "$2" || -z "$3" ]]; then
            log_error "Table names are required for full deployment"
            show_usage
            exit 1
        fi
        check_prerequisites
        build_image
        push_image
        update_pipeline_config
        deploy_k8s_resources
        compile_pipeline "$2" "$3"
        log_info "Full deployment completed successfully!"
        log_info "Upload the generated pipeline YAML to your Kubeflow Pipelines UI"
        ;;
    "help"|"--help"|"-h"|"")
        show_usage
        ;;
    *)
        log_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac