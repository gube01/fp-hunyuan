# Minimal Dockerfile - Download models at runtime instead
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models

# Create app directory
WORKDIR /app
RUN mkdir -p /app/models

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy repository files
COPY . /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install latest diffusers from git (needed for HunyuanVideo)
RUN pip install git+https://github.com/huggingface/diffusers.git

# Install additional dependencies
RUN pip install accelerate xformers opencv-python pillow

# Test that everything imports correctly
RUN python -c "import torch; from diffusers import HunyuanVideoPipeline; print('✓ All imports successful')"

# Create startup script
RUN echo '#!/bin/bash\n\
echo "HunyuanVideo Environment Ready!"\n\
echo "GPU Status:"\n\
nvidia-smi\n\
echo "Models will be downloaded on first use"\n\
exec "$@"' > /app/start.sh && chmod +x /app/start.sh

ENTRYPOINT ["/app/start.sh"]
CMD ["python", "-c", "print('Ready to run HunyuanVideo scripts!')"]
