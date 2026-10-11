FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace/cgcwra

# Install Python dependencies (we install dependencies first to leverage Docker cache)
COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -e .

# We don't copy the rest of the code here to avoid rebuilding the image 
# every time the code changes. Instead, we mount the codebase in docker-compose.
# But for standalone usage, one could add: COPY . .

CMD ["/bin/bash"]
