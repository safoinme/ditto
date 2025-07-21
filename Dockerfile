# Use Kubeflow Jupyter PyTorch CUDA Full image as base
FROM kubeflownotebookswg/jupyter-pytorch-cuda-full:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VERSION=12.2

# Install system dependencies
USER root

# Add NVIDIA CUDA repository and install CUDA 12.2 toolkit
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    build-essential \
    gnupg2 \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get install -y cuda-toolkit-12-2 \
    && rm -rf /var/lib/apt/lists/* cuda-keyring_1.0-1_all.deb

# Install uv for fast Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Create directories for jovyan user with proper ownership
RUN mkdir -p /home/jovyan/.local/bin /home/jovyan/.config/uv && chown -R jovyan:users /home/jovyan/.local /home/jovyan/.config

# Switch back to jovyan user (standard for Kubeflow notebooks)
USER jovyan

# Add uv to jovyan user's PATH
RUN echo 'export PATH="/home/jovyan/.local/bin:$PATH"' >> /home/jovyan/.bashrc
ENV PATH="/home/jovyan/.local/bin:$PATH"

# Install uv for jovyan user
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set working directory
WORKDIR /home/jovyan

# Copy requirements file
COPY --chown=jovyan:users requirements.txt .

# Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.2 drivers)  
RUN /home/jovyan/.local/bin/uv pip install --system \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining requirements using uv for fast installation
RUN /home/jovyan/.local/bin/uv pip install --system -r requirements.txt

# Verify installations
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"