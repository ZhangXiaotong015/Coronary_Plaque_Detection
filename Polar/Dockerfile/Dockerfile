# --------------------------------------------------------
# 1. Base image
# --------------------------------------------------------
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# --------------------------------------------------------
# 2. Avoid interactive prompts during apt install
# --------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive

# --------------------------------------------------------
# 3. Install system dependencies
# --------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    vim \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------
# 4. Copy requirements and install Python deps
# --------------------------------------------------------
WORKDIR /app

COPY requirements.txt /app/

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------
# 5. Copy inference script and model
# --------------------------------------------------------
COPY inference.py /app/inference.py
COPY ct2polar.py /app/ct2polar.py
COPY Block25D_Polar_Auto.py /app/Block25D_Polar_Auto.py
COPY spread_out_view.py /app/spread_out_view.py
COPY model/ /app/model/
COPY datasets/ /app/datasets/

# --------------------------------------------------------
# 6. Define default environment variables
# --------------------------------------------------------
ENV MODEL_PATH=/app/model/40_checkpoint.npz
ENV SPARSE_MATRIX_PATH=/app/model/lumenCenter_sampleLine_sparseMetrix.npy
ENV SAMPLE_LINE_PATH=/app/model/lumenCenter_sampleLine.npy

# --------------------------------------------------------
# 7. Default entrypoint (you can override when running)
# --------------------------------------------------------
ENTRYPOINT ["python", "/app/inference.py"]
# CMD ["python", "/app/inference.py"]

