# Use NVIDIA CUDA 12.2 toolkit as base image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV BERT_MODEL_PATH=/home/jovyan/models/bert-base-uncased
ENV NLTK_DATA=/home/jovyan/nltk_data

# Install system dependencies including Python and Jupyter
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    git \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Install Jupyter and create jovyan user
RUN pip3 install --no-cache-dir jupyter jupyterlab \
    && useradd -m -s /bin/bash jovyan \
    && usermod -aG sudo jovyan

# Install uv for fast Python package management and set up path properly
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.2)
RUN uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Copy requirements file and install remaining requirements (as root)
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Install spaCy model (as root)
RUN python3 -m spacy download en_core_web_sm

# Create directories for jovyan user with proper ownership
RUN mkdir -p /home/jovyan/.local/bin /home/jovyan/.config/uv && chown -R jovyan:users /home/jovyan/.local /home/jovyan/.config

# Switch to jovyan user
USER jovyan

# Set working directory
WORKDIR /home/jovyan

# Add uv to jovyan user's PATH
RUN echo 'export PATH="/home/jovyan/.local/bin:$PATH"' >> /home/jovyan/.bashrc
ENV PATH="/home/jovyan/.local/bin:$PATH"

# Install uv for jovyan user
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy the local directory contents into the container
COPY --chown=jovyan:users . .

# Download models and NLTK data (run as jovyan user)
RUN python3 download_bert_model.py

# Verify installations
RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"