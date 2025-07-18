# Use Kubeflow Jupyter PyTorch CUDA Full image as base
FROM kubeflownotebookswg/jupyter-pytorch-cuda-full:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VERSION=12.2

# Install system dependencies and CUDA 12.2 toolkit
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

# Switch back to jovyan user (standard for Kubeflow notebooks)
USER jovyan

# Add uv to jovyan user's PATH
RUN echo 'export PATH="/home/jovyan/.cargo/bin:$PATH"' >> /home/jovyan/.bashrc
ENV PATH="/home/jovyan/.cargo/bin:$PATH"

# Install uv for jovyan user
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set working directory
WORKDIR /home/jovyan

# Copy requirements file
COPY --chown=jovyan:users requirements.txt .

# Install PyTorch with CUDA 12.2 support first using uv
RUN /home/jovyan/.cargo/bin/uv pip install --system \
    torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install remaining requirements using uv for fast installation
RUN /home/jovyan/.cargo/bin/uv pip install --system \
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
    nltk> \
    tensorboard \
    tensorboardX

# Verify installations
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Set default command
CMD ["start-notebook.sh"] 