# syntax=docker/dockerfile:1.7

# ─────────────────────────────────────────────────────────────────────────────
# TrainStudio — CUDA runtime image
#
# Built for an NVIDIA workstation. Adjust CUDA_IMAGE and TORCH_INDEX_URL to
# match the driver on the host — see the README.
# CUDA 12.4 + cuDNN, Python 3.11, PyTorch from the cu124 wheel index.
#
#   docker compose build
#   docker compose up -d
# ─────────────────────────────────────────────────────────────────────────────

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

# The PyTorch wheel index. Override to cu121/cu128 for a different driver, or to
# "https://pypi.org/simple" for a CPU-only build.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
ARG PYTHON_VERSION=3.11

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── System packages ──────────────────────────────────────────────────────────
# software-properties-common brings in add-apt-repository, which we need for the
# deadsnakes PPA (Ubuntu 22.04 ships Python 3.10 by default).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
        git \
        gosu \
        tini \
        # OpenCV / matplotlib / SimpleITK runtime libraries
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ── A virtualenv, so the image never fights the system Python ────────────────
ENV VIRTUAL_ENV=/opt/venv
RUN python${PYTHON_VERSION} -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
RUN pip install --upgrade pip setuptools wheel

# ── PyTorch first, from the CUDA wheel index ─────────────────────────────────
# Installed before requirements.txt so that pip resolves the rest against this
# exact build instead of pulling a generic PyPI torch on top of it.
RUN pip install \
        --index-url ${TORCH_INDEX_URL} \
        torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION}

# ── The remaining dependencies ───────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# ── The application ──────────────────────────────────────────────────────────
WORKDIR /app
COPY . /app

# Directories that are mounted as volumes at run time. They are created here so
# the image works even when nothing is mounted.
# The state dir lives outside /home so that bind-mounting the host's /home
# cannot shadow it.
RUN mkdir -p /cache /opt/trainstudio-state \
    && useradd --create-home --home-dir /opt/trainstudio-home --shell /bin/bash \
               --uid 1000 trainstudio \
    && chown -R trainstudio:trainstudio /app /cache /opt/trainstudio-state /opt/trainstudio-home

# ── Runtime configuration ────────────────────────────────────────────────────
ENV TRAINSTUDIO_HOME=/opt/trainstudio-state \
    # Where the pretrained weights downloaded by timm/HF/Ultralytics are cached
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache \
    MPLCONFIGDIR=/cache/matplotlib \
    YOLO_CONFIG_DIR=/cache/ultralytics \
    # Streamlit: no file watcher, no run-on-save, listen on every interface
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_SERVER_RUN_ON_SAVE=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    # Keep the BLAS threads from fighting the DataLoader workers
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# tini reaps the training subprocesses that runner.py leaves behind
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["streamlit", "run", "app.py"]
