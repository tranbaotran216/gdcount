FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV CUDA_HOME=/usr/local/cuda

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ninja-build \
    libglib2.0-0 \
    libgl1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY GroundingDINO /app/GroundingDINO

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -e /app/GroundingDINO --no-build-isolation

RUN pip install --no-cache-dir \
    numpy==2.2.6 \
    scipy==1.15.3 \
    pandas==2.3.3 \
    matplotlib==3.10.8 \
    opencv-python-headless==4.12.0.88 \
    streamlit==1.52.1 \
    streamlit-drawable-canvas-fix \
    transformers==4.57.3 \
    tokenizers==0.22.1 \
    huggingface-hub==0.36.0 \
    timm==1.0.22 \
    einops==0.8.1 \
    pycocotools==2.0.10 \
    supervision==0.27.0 \
    yapf==0.43.0 \
    pyyaml \
    tqdm \
    pillow \
    gitpython \
    addict

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
