FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# --- Env & OS deps ---
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv python3-dev \
      build-essential git curl ca-certificates ffmpeg \
      libglib2.0-0 libgl1 libxext6 libsm6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# --- Install Python deps as root (system site-packages) ---
RUN python3 -m pip install --upgrade pip wheel

# GPU=1 => CUDA wheels; GPU=0 => CPU wheels (for dev)
ARG GPU=1
RUN if [ "$GPU" = "1" ]; then \
      python3 -m pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision ; \
    else \
      python3 -m pip install torch torchvision ; \
    fi

# Copy requirements and install dependencies
COPY requirements.txt /tmp/requirements.txt

# Install base dependencies with version constraints from requirements.txt
# Use opencv-python-headless instead of opencv-python for Docker
# The sed command replaces opencv-python with opencv-python-headless for headless environments
RUN sed 's/opencv-python>=4\.5\.0/opencv-python-headless>=4.5.0/g' /tmp/requirements.txt > /tmp/requirements-docker.txt && \
    python3 -m pip install -r /tmp/requirements-docker.txt && \
    python3 -m pip install tifffile && \
    rm /tmp/requirements.txt /tmp/requirements-docker.txt

# Verify numpy version is <2.0 to avoid conflicts
RUN python3 -c "import numpy as np; version=tuple(map(int, np.__version__.split('.')[:2])); assert version[0] < 2, f'NumPy {np.__version__} >= 2.0 will cause conflicts with matplotlib, pandas, and contourpy. Please use numpy<2.0'; print(f'✓ NumPy {np.__version__} is compatible')"

# Sanity print and comprehensive verification
RUN python3 - <<'PY'
import sys
print(f"Python: {sys.version}")

import torch, cellpose, numpy, matplotlib, cv2
print(f"✓ Torch: {torch.__version__}, CUDA build: {torch.version.cuda}, CUDA available: {torch.cuda.is_available()}")
print(f"✓ Cellpose: {getattr(cellpose, '__version__', 'unknown')}")
print(f"✓ NumPy: {numpy.__version__}")
print(f"✓ Matplotlib: {matplotlib.__version__}")
print(f"✓ OpenCV: {cv2.__version__}")

# Verify numpy version constraint
major, minor = map(int, numpy.__version__.split('.')[:2])
if major >= 2:
    print(f"⚠ WARNING: NumPy {numpy.__version__} may cause conflicts!")
    sys.exit(1)
else:
    print(f"✓ NumPy version constraint satisfied (<2.0)")
PY

# --- Create non-root user AFTER installs ---
RUN useradd -m -u 1000 appuser
WORKDIR /workspace
USER appuser

# Default entrypoint (override in your job)
CMD ["python3", "-m", "cellpose", "--help"]