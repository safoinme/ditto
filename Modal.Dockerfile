# Use a simpler CUDA base image without s6-overlay for Modal compatibility
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VERSION=12.2
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create jovyan user (common convention for Jupyter containers)
RUN useradd -m -s /bin/bash jovyan
USER jovyan

# Set working directory
WORKDIR /home/jovyan

# Install uv for jovyan user
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/jovyan/.local/bin:$PATH"

# Copy requirements file
COPY --chown=jovyan:users requirements.txt .

# Install PyTorch with CUDA 12.2 support first using uv
RUN /home/jovyan/.local/bin/uv pip install --system \
    torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install Jupyter and remaining requirements using uv for fast installation
RUN /home/jovyan/.local/bin/uv pip install --system \
    jupyter \
    jupyterlab \
    notebook \
    gensim \
    numpy \
    regex \
    scipy \
    sentencepiece \
    scikit-learn \
    spacy \
    transformers \
    tqdm \
    jsonlines \
    nltk \
    tensorboard \
    tensorboardX

# Verify installations
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Copy the entire project (done last to optimize Docker layer caching)
COPY --chown=jovyan:users . .

# Set default command for Modal compatibility (no s6-overlay)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]