# Ditto Docker Setup for CUDA 12.2

This Docker setup provides a containerized environment for the Ditto entity matching project with CUDA 12.2 support, optimized for production environments with NVIDIA CUDA 12.2.

## Base Image

- **Base**: `kubeflownotebookswg/jupyter-pytorch-cuda-full:latest`
- **CUDA Version**: 12.2
- **PyTorch Version**: 2.1.0 with CUDA 12.1 support (compatible with CUDA 12.2)
- **Package Manager**: `uv` for fast Python package installation

## Features

- ✅ CUDA 12.2 compatible PyTorch installation
- ✅ All project requirements pre-installed
- ✅ Optimized for Kubeflow/Jupyter environments
- ✅ Fast package installation using `uv`
- ✅ Non-root user setup (jovyan)
- ✅ JupyterLab ready

## Building the Container

### Prerequisites

- Docker installed on your system
- NVIDIA Docker runtime (for GPU support)
- NVIDIA GPU with CUDA 12.2 support

### Build Command

```bash
# Build the Docker image
docker build -t ditto-cuda12.2:latest .

# Build with custom tag
docker build -t your-registry/ditto-cuda12.2:v1.0 .
```

### Build Arguments (Optional)

The Dockerfile accepts the following build arguments:

```bash
# Example with build arguments
docker build \
  --build-arg CUDA_VERSION=12.2 \
  -t ditto-cuda12.2:latest .
```

## Running the Container

### Basic Run (CPU only)

```bash
docker run -it --rm \
  -p 8888:8888 \
  -v $(pwd):/home/jovyan/work \
  ditto-cuda12.2:latest
```

### GPU-enabled Run

```bash
# With NVIDIA Docker runtime
docker run -it --rm \
  --gpus all \
  -p 8888:8888 \
  -v $(pwd):/home/jovyan/work \
  ditto-cuda12.2:latest

# Or with specific GPU
docker run -it --rm \
  --gpus '"device=0"' \
  -p 8888:8888 \
  -v $(pwd):/home/jovyan/work \
  ditto-cuda12.2:latest
```

### Development Setup

```bash
# Mount entire project directory and expose additional ports
docker run -it --rm \
  --gpus all \
  -p 8888:8888 \
  -p 6006:6006 \
  -v $(pwd):/home/jovyan/work \
  -e JUPYTER_ENABLE_LAB=yes \
  ditto-cuda12.2:latest
```

## Verification

After starting the container, you can verify the installation:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Test GPU access
if torch.cuda.is_available():
    x = torch.randn(3, 3).cuda()
    print(f"GPU tensor: {x}")
```

## Kubeflow Deployment

### PodDefault for GPU

```yaml
apiVersion: kubeflow.org/v1alpha1
kind: PodDefault
metadata:
  name: add-gpu
  namespace: your-namespace
spec:
  selector:
    matchLabels:
      add-gpu: "true"
  desc: "Add GPU support"
  env:
  - name: NVIDIA_VISIBLE_DEVICES
    value: all
  - name: NVIDIA_DRIVER_CAPABILITIES
    value: compute,utility
  resources:
    limits:
      nvidia.com/gpu: 1
```

### Notebook Server Configuration

```yaml
apiVersion: kubeflow.org/v1
kind: Notebook
metadata:
  name: ditto-notebook
  namespace: your-namespace
spec:
  template:
    spec:
      containers:
      - name: notebook
        image: your-registry/ditto-cuda12.2:v1.0
        resources:
          limits:
            nvidia.com/gpu: 1
            cpu: "4"
            memory: "16Gi"
          requests:
            cpu: "2"
            memory: "8Gi"
        volumeMounts:
        - name: workspace
          mountPath: /home/jovyan/work
      volumes:
      - name: workspace
        persistentVolumeClaim:
          claimName: ditto-workspace
```

## Package Details

The container includes the following packages optimized for CUDA 12.2:

### Core ML Packages
- **PyTorch**: 2.1.0+cu121 (CUDA 12.1 compatible with 12.2)
- **TorchVision**: 0.16.0+cu121
- **TorchAudio**: 2.1.0+cu121

### NLP & Text Processing
- **Transformers**: >=4.21.0
- **SpaCy**: >=3.4
- **SentencePiece**
- **NLTK**: >=3.7
- **Gensim**

### Scientific Computing
- **NumPy**
- **SciPy**
- **scikit-learn**

### Utilities
- **TensorBoard & TensorboardX**
- **TQDM**: >=4.64.0
- **JSONLines**: >=2.0.0
- **Regex**

## Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**
   ```bash
   # Check CUDA version in container
   nvidia-smi
   nvcc --version
   ```

2. **GPU Not Accessible**
   ```bash
   # Verify NVIDIA Docker runtime
   docker info | grep nvidia
   
   # Check GPU access
   docker run --rm --gpus all nvidia/cuda:12.2-base-ubuntu22.04 nvidia-smi
   ```

3. **Permission Issues**
   ```bash
   # The container runs as jovyan user (UID 1000)
   # Ensure your host directories have appropriate permissions
   sudo chown -R 1000:1000 /path/to/your/data
   ```

4. **Memory Issues**
   ```bash
   # Increase Docker memory limit
   # Check Docker Desktop settings or system configuration
   ```

### Performance Optimization

1. **Use SSD storage** for volume mounts
2. **Allocate sufficient memory** (recommended: 16GB+)
3. **Use GPU memory management** in your code:
   ```python
   torch.cuda.empty_cache()  # Clear GPU memory
   ```

## Security Considerations

- Container runs as non-root user (`jovyan`)
- Only necessary system packages installed
- Use specific image tags in production
- Regularly update base images for security patches

## Support

For issues related to:
- **CUDA compatibility**: Check NVIDIA driver version compatibility
- **PyTorch issues**: Refer to PyTorch documentation
- **Kubeflow integration**: Check Kubeflow documentation
- **Container issues**: Review Docker logs with `docker logs <container_id>`

## License

This Dockerfile configuration follows the same license as the Ditto project. 