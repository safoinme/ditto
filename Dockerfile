# Streamlined Dockerfile for Ditto Entity Matching with PyTorch CUDA 12.2
# Focused on Ditto + PyHive integration without Spark overhead
FROM nvidia/cuda:12.2-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VERSION=12.2
ENV PYTHONPATH=/app:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Essential build tools
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    # Python and development
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    # For Hive/Hadoop connectivity  
    libsasl2-dev \
    libsasl2-modules \
    libsasl2-modules-gssapi-mit \
    libkrb5-dev \
    # System libraries for ML/DL
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    # Networking and security
    ca-certificates \
    gnupg \
    lsb-release \
    # Utilities
    vim \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip and install build tools
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Create application structure
RUN mkdir -p /app/ditto \
    && mkdir -p /data/input \
    && mkdir -p /data/output \
    && mkdir -p /checkpoints \
    && mkdir -p /models \
    && mkdir -p /config

# Set working directory
WORKDIR /app/ditto

# Copy requirements files first for better caching
COPY requirements.txt ./requirements_ditto.txt

# Create streamlined requirements file (no Spark!)
RUN cat > requirements_complete.txt << 'EOF'
# PyTorch with CUDA 12.2 support (will be installed separately)
# torch==2.1.2+cu121
# torchvision==0.16.2+cu121
# torchaudio==2.1.2+cu121

# Core Ditto ML/DL requirements
gensim>=4.3.0
numpy>=1.21.0
regex>=2023.0.0
scipy>=1.9.0
sentencepiece>=0.1.99
scikit-learn>=1.3.0
spacy>=3.7.0
transformers>=4.36.0
tqdm>=4.65.0
jsonlines>=3.1.0
nltk>=3.8.0
tensorboard>=2.15.0
tensorboardX>=2.6.0

# Data processing and analytics
pandas==2.0.3
pyarrow==15.0.0

# Hive connectivity (no Spark needed!)
pyhive==0.7.0
thrift==0.16.0
thrift-sasl==0.4.3
pure-sasl==0.6.2
sasl>=0.3.1

# Entity matching specific
SQLAlchemy==2.0.36

# Kubeflow and orchestration
kfp==2.7.0
kubernetes==29.0.0

# Development and utilities
pydantic==2.6.1
python-dotenv==1.0.1
click>=8.1.0
PyYAML>=6.0
requests>=2.28.0

# Testing (optional)
pytest==8.0.0
pytest-cov==4.1.0

# Code quality (optional)
black==24.1.1
isort==5.13.2
mypy==1.8.0
EOF

# Install PyTorch with CUDA 12.2 support first
RUN pip install --no-cache-dir \
    torch==2.1.2+cu121 \
    torchvision==0.16.2+cu121 \
    torchaudio==2.1.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install all other requirements
RUN pip install --no-cache-dir -r requirements_complete.txt

# Install spaCy English model
RUN python -m spacy download en_core_web_lg

# Verify CUDA and PyTorch installation
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'cuDNN version: {torch.backends.cudnn.version()}')"

# Verify Hive connectivity components (no Spark!)
RUN python -c "import pyhive; from pyhive import hive; print('PyHive: OK')" && \
    python -c "import thrift; print('Thrift: OK')" && \
    python -c "import sasl; print('SASL: OK')"

# Verify ML/DL components
RUN python -c "import transformers; print(f'Transformers version: {transformers.__version__}')" && \
    python -c "import spacy; print(f'spaCy version: {spacy.__version__}')" && \
    python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')" && \
    python -c "import pandas; print(f'Pandas version: {pandas.__version__}')"

# Verify Kubeflow components
RUN python -c "import kfp; print(f'Kubeflow Pipelines SDK version: {kfp.__version__}')"

# Copy the entire ditto project
COPY . .

# Create enhanced entrypoint script
RUN cat > entrypoint.sh << 'EOF'
#!/bin/bash
set -e

echo "=============================================="
echo "🚀 Ditto Entity Matching Container"
echo "=============================================="

echo "Environment Information:"
echo "- PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "- CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "- CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
    echo "- GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
    echo "- GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
fi
echo "- Working directory: $(pwd)"
echo "- PyHive available: $(python -c 'import pyhive; print("Yes")' 2>/dev/null || echo "No")"

echo "Available files:"
ls -la

echo "=============================================="

# Execute the command passed to docker run
if [ $# -eq 0 ]; then
    echo "No command specified. Starting interactive bash shell..."
    echo "Try: python matcher.py --help"
    echo "Or: python hive_data_extractor.py --help"
    exec /bin/bash
else
    echo "Executing command: $@"
    exec "$@"
fi
EOF

RUN chmod +x entrypoint.sh

# Create helpful aliases
RUN echo 'alias ll="ls -la"' >> /root/.bashrc && \
    echo 'alias python="python3"' >> /root/.bashrc && \
    echo 'export PATH="/app/ditto:$PATH"' >> /root/.bashrc

# Set final working directory
WORKDIR /app/ditto

# Expose ports (remove Spark ports)
EXPOSE 8888

# Health check (no Spark imports!)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch, pyhive, transformers, kfp; print('Health check passed')" || exit 1

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]

# Default command
CMD ["python", "--version"]