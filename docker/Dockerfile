ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV MPLBACKEND=Agg
ENV HF_HOME=/workspace/cgcwra/artifacts/hf_cache
ENV TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    iputils-ping \
    iproute2 \
    dnsutils \
    netcat-openbsd \
    procps \
    tini \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/cgcwra

COPY pyproject.toml README.md requirements.txt ./
COPY requirements ./requirements
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

RUN python -m pip install --upgrade pip && \
    pip install -r requirements/dev.txt && \
    pip install -e .

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["/bin/bash"]
