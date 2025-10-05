# Use prebuilt Torch + CUDA 11.8 runtime (no giant wheel download)
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# System deps for rasterio/GDAL/OpenCV + utils
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev libproj-dev libgeos-dev libspatialindex-dev \
    build-essential cmake pkg-config \
    git wget curl ca-certificates unzip \
    libhdf5-dev libnetcdf-dev \
    libopencv-dev libjpeg-dev libpng-dev libtiff-dev \
    libgeotiff-dev libboost-all-dev bzip2 \
 && rm -rf /var/lib/apt/lists/*

# (Optional) AWS CLI v2 — handy for quick sts/s3 tests
RUN curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip" \
 && unzip /tmp/awscliv2.zip -d /tmp \
 && /tmp/aws/install \
 && rm -rf /tmp/aws /tmp/awscliv2.zip

# Make Python chatty and avoid .pyc
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install Python deps with constraints (avoid NumPy 2.x ABI issues)
COPY constraints.txt requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install -c constraints.txt "numpy<2" "scipy<1.11" && \
    pip install -c constraints.txt -r requirements.txt

# App code
COPY . .

# Quick sanity (fail fast if something is off)
RUN python -c "import torch, rasterio, cv2; \
print('Sanity OK:', 'torch', torch.__version__, 'cuda', torch.version.cuda, \
'cuda_available', torch.cuda.is_available(), 'rasterio', rasterio.__version__, 'cv2', cv2.__version__)"

# Default entry (override in compose if you like)
CMD ["python", "main.py"]
