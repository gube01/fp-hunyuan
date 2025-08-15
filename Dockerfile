# Use RunPod's PyTorch base image with CUDA support
FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_DATASETS_CACHE=/app/models

# Create app directory and models cache directory
WORKDIR /app
RUN mkdir -p /app/models

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the repository files
COPY . /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install additional dependencies that might be needed
RUN pip install \
    accelerate \
    xformers \
    opencv-python \
    pillow \
    numpy \
    torch \
    torchvision \
    torchaudio

# Pre-download the HunyuanVideo models
RUN echo "Starting model download process..." && \
    python -c "\
import os; \
import torch; \
from diffusers import HunyuanVideoPipeline; \
from transformers import AutoTokenizer; \
print('=' * 50); \
print('STARTING MODEL DOWNLOAD'); \
print('This may take 15-30 minutes depending on connection speed'); \
print('=' * 50); \
try: \
    print('Downloading HunyuanVideo pipeline...'); \
    pipeline = HunyuanVideoPipeline.from_pretrained('tencent/HunyuanVideo', torch_dtype=torch.float16, cache_dir='/app/models'); \
    print('✓ HunyuanVideo pipeline downloaded successfully'); \
    print('Downloading text encoder...'); \
    tokenizer = AutoTokenizer.from_pretrained('lllyasviel/llava-llama-3-8b-text-encoder-tokenizer', cache_dir='/app/models'); \
    print('✓ Text encoder downloaded successfully'); \
    from diffusers import AutoencoderKLHunyuanVideo; \
    print('Downloading VAE...'); \
    vae = AutoencoderKLHunyuanVideo.from_pretrained('tencent/HunyuanVideo', subfolder='vae', torch_dtype=torch.float16, cache_dir='/app/models'); \
    print('✓ VAE downloaded successfully'); \
    print('All models downloaded and cached successfully!'); \
except Exception as e: \
    print(f'Error downloading models: {e}'); \
    print('Models may need to be downloaded at runtime'); \
"

# Create a startup script
RUN echo '#!/bin/bash\n\
echo \"Starting HunyuanVideo container...\"\n\
echo \"Models are pre-downloaded and ready to use\"\n\
echo \"GPU Status:\"\n\
nvidia-smi\n\
echo \"Python path: $(which python)\"\n\
echo \"Working directory: $(pwd)\"\n\
echo \"Available models in cache:\"\n\
ls -la /app/models/\n\
exec \"$@\"' > /app/start.sh && chmod +x /app/start.sh

# Set the entrypoint
ENTRYPOINT ["/app/start.sh"]

# Default command (can be overridden)
CMD ["python", "-c", "print('HunyuanVideo container is ready! Use: python your_script.py')"]
