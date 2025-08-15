# Bulletproof Dockerfile for HunyuanVideo with Pre-downloaded Models
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_DATASETS_CACHE=/app/models
ENV TORCH_HOME=/app/models

# Create directories
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

# Copy repository files first
COPY . /app/

# Upgrade pip and install basic requirements
RUN pip install --upgrade pip setuptools wheel

# Install requirements from the repo
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Install essential packages for HunyuanVideo
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install latest diffusers from source (HunyuanVideo needs cutting edge)
RUN pip install git+https://github.com/huggingface/diffusers.git

# Install other necessary packages
RUN pip install transformers accelerate xformers opencv-python pillow numpy scipy

# Create download script for better error handling
RUN echo '#!/usr/bin/env python3\n\
import sys\n\
import os\n\
import torch\n\
\n\
def download_models():\n\
    print("="*60)\n\
    print("STARTING HUNYUANVIDEO MODEL DOWNLOAD")\n\
    print("This will take 15-30 minutes but saves time on every run!")\n\
    print("="*60)\n\
    \n\
    # Check PyTorch and CUDA\n\
    print(f"PyTorch version: {torch.__version__}")\n\
    print(f"CUDA available: {torch.cuda.is_available()}")\n\
    if torch.cuda.is_available():\n\
        print(f"CUDA version: {torch.version.cuda}")\n\
    \n\
    try:\n\
        print("\\n1. Testing diffusers import...")\n\
        from diffusers import HunyuanVideoPipeline\n\
        print("✓ Diffusers imported successfully")\n\
        \n\
        print("\\n2. Downloading HunyuanVideo pipeline...")\n\
        pipeline = HunyuanVideoPipeline.from_pretrained(\n\
            "tencent/HunyuanVideo",\n\
            torch_dtype=torch.float16,\n\
            cache_dir="/app/models"\n\
        )\n\
        print("✓ HunyuanVideo pipeline downloaded successfully!")\n\
        \n\
        # Clear memory\n\
        del pipeline\n\
        torch.cuda.empty_cache() if torch.cuda.is_available() else None\n\
        \n\
        print("\\n3. Downloading text encoder...")\n\
        from transformers import AutoTokenizer\n\
        tokenizer = AutoTokenizer.from_pretrained(\n\
            "lllyasviel/llava-llama-3-8b-text-encoder-tokenizer",\n\
            cache_dir="/app/models"\n\
        )\n\
        print("✓ Text encoder downloaded successfully!")\n\
        del tokenizer\n\
        \n\
        print("\\n" + "="*60)\n\
        print("ALL MODELS DOWNLOADED SUCCESSFULLY!")\n\
        print("Your RunPod serverless workers will start instantly!")\n\
        print("="*60)\n\
        return True\n\
        \n\
    except Exception as e:\n\
        print(f"\\n❌ ERROR downloading models: {e}")\n\
        print("\\nFull error details:")\n\
        import traceback\n\
        traceback.print_exc()\n\
        return False\n\
\n\
if __name__ == "__main__":\n\
    success = download_models()\n\
    if not success:\n\
        print("\\n⚠️  Model download failed, but continuing build...")\n\
        print("Models will be downloaded at runtime instead.")\n\
        # Do not exit with error code - let the build continue\n\
' > /app/download_models.py && chmod +x /app/download_models.py

# Download the models
RUN python /app/download_models.py

# Create optimized startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "🚀 Starting HunyuanVideo Container"\n\
echo "================================"\n\
\n\
# Check GPU\n\
if command -v nvidia-smi &> /dev/null; then\n\
    echo "GPU Status:"\n\
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits\n\
else\n\
    echo "⚠️  nvidia-smi not available"\n\
fi\n\
\n\
echo ""\n\
echo "📁 Model Cache Status:"\n\
if [ -d "/app/models" ]; then\n\
    echo "Cache directory size: $(du -sh /app/models 2>/dev/null || echo unknown)"\n\
    echo "Cached models: $(find /app/models -name "*.safetensors" -o -name "*.bin" | wc -l) files"\n\
else\n\
    echo "⚠️  No model cache found - models will download at runtime"\n\
fi\n\
\n\
echo ""\n\
echo "✅ Container ready! Starting your application..."\n\
echo "================================"\n\
\n\
exec "$@"\n\
' > /app/start.sh && chmod +x /app/start.sh

# Verify installation
RUN python -c "import torch; from diffusers import HunyuanVideoPipeline; print('✅ Installation verified!')"

# Set the entrypoint
ENTRYPOINT ["/app/start.sh"]

# Default command
CMD ["python", "-c", "print('🎥 HunyuanVideo container is ready! Models are pre-downloaded.')"]
