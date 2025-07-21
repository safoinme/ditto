# Use Kubeflow Jupyter PyTorch CUDA Full image as base
FROM kubeflownotebookswg/jupyter-pytorch-cuda-full:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VERSION=12.4

# Install system dependencies and CUDA 12.4 toolkit
USER root

# Update package lists and install essential tools
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Create .local/bin directory for jovyan user with proper ownership
RUN mkdir -p /home/jovyan/.local/bin && chown -R jovyan:users /home/jovyan/.local

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

# Install PyTorch with CUDA 12.4 support first using uv
RUN /home/jovyan/.local/bin/uv pip install --system \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install remaining requirements using uv for fast installation
RUN /home/jovyan/.local/bin/uv pip install --system -r requirements.txt

# Verify installations
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"