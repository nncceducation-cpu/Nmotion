# Nmotion — neonatal movement analysis web app (CPU by default)
FROM python:3.11-slim

# System libs: ffmpeg + OpenGL/glib for OpenCV video I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (much smaller than the default CUDA wheels).
# For GPU, see the build-arg override in README.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX} \
        torch>=2.4.0 torchvision>=0.19.0

# Remaining pipeline + web deps
COPY requirements.txt webapp/requirements-web.txt ./
RUN pip install --no-cache-dir \
        antropy scipy numpy pandas scikit-learn matplotlib \
        opencv-python-headless xgboost \
    && pip install --no-cache-dir -r requirements-web.txt

# App code
COPY pipeline ./pipeline
COPY webapp ./webapp
COPY run.py ./run.py
COPY models ./models
COPY train.py ./train.py

ENV NMOTION_MAX_FRAMES=240 \
    NMOTION_MAX_WIDTH=640 \
    NMOTION_DATA_DIR=/app/webapp/data_runtime \
    TORCH_HOME=/app/.torch \
    PYTHONUNBUFFERED=1

EXPOSE 8000
# Shell form so cloud hosts (Render/Railway/Fly) can inject $PORT; falls back to 8000.
CMD uvicorn webapp.app:app --host 0.0.0.0 --port ${PORT:-8000}
