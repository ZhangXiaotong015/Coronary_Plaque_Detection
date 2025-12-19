# --------------------------------------------------------
# 1. Use CUDA 11.2 + cuDNN 8 base image (TF 2.6 compatible)
# --------------------------------------------------------
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# --------------------------------------------------------
# 2. Avoid interactive prompts during apt install
# --------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive

# --------------------------------------------------------
# 3. Install system dependencies
# --------------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Use python3 as python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# --------------------------------------------------------
# 4. Copy requirements and install Python deps
# --------------------------------------------------------
WORKDIR /app

COPY requirements.txt /app/

RUN pip3 install --no-cache-dir -r requirements.txt

# --------------------------------------------------------
# 5. Copy inference script and model
# --------------------------------------------------------
COPY inference.py /app/inference.py
COPY spread_out_view.py /app/spread_out_view.py
COPY ct2polar.py /app/ct2polar.py
COPY Block25D_enlarged_Auto.py /app/Block25D_enlarged_Auto.py
COPY model/ /app/model/

# --------------------------------------------------------
# 6. Define default environment variables
# --------------------------------------------------------
# Output directory inside container
ENV MODEL_PATH=/app/model/model_041.hdf5
ENV SPARSE_MATRIX_PATH=/app/model/lumenCenter_sampleLine_sparseMetrix.npy
ENV SAMPLE_LINE_PATH=/app/model/lumenCenter_sampleLine.npy

# --------------------------------------------------------
# 7. Create output folder
# --------------------------------------------------------
RUN mkdir -p /output

# --------------------------------------------------------
# 8. Default entrypoint (you can override when running)
# --------------------------------------------------------
ENTRYPOINT ["python", "/app/inference.py"]
# CMD ["python", "/app/inference.py"]

