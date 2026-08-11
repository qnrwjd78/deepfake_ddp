FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        git \
        curl \
        ca-certificates \
        tmux \
        ffmpeg \
        zip \
        unzip \
        p7zip-full \
        xz-utils \
        zstd \
        pigz \
        pbzip2 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0+cu124 \
        torchvision==0.21.0+cu124 \
    && python -m pip install \
        huggingface_hub[cli] \
        hf_transfer \
        accelerate \
        transformers \
        safetensors

CMD ["/bin/bash"]
