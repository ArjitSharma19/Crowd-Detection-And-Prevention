# Use official PyTorch runtime with CUDA support
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies needed by OpenCV, UI libraries, and compiling packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file first for layer caching
COPY requirements.txt /app/

# Install Python dependencies. 
# We filter out torch and torchvision from requirements.txt to avoid re-downloading 
# them since they are already prepackaged in the base CUDA container image.
RUN grep -v -E "torch|torchvision" requirements.txt > temp-reqs.txt && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r temp-reqs.txt && \
    rm temp-reqs.txt

# Copy project files
COPY src/ /app/src/
COPY webapp/ /app/webapp/
COPY models/ /app/models/
COPY custom_bytetrack.yaml /app/
COPY main.py /app/

# Expose FastAPI port
EXPOSE 8000

# Start command running the main server
CMD ["python", "main.py"]
