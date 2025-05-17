# Stage 1: Build environment
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS builder

# Install Miniconda
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

# Add conda to path
ENV PATH="/opt/conda/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy only environment definition first
COPY environment.yaml .

# Create conda environment with minimal packages
RUN conda env create -f environment.yaml && \
    conda run -n pano3dc pip install timm==1.0.15 controlnet-aux==0.0.9 --no-deps && \
    conda clean -afy && \
    find /opt/conda -follow -type f -name '*.a' -delete && \
    find /opt/conda -follow -type f -name '*.js.map' -delete && \
    find /opt/conda -name '__pycache__' -type d -exec rm -rf {} + || true

# Stage 2: Runtime environment
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy conda environment from builder stage
COPY --from=builder /opt/conda /opt/conda

# Add conda to path
ENV PATH="/opt/conda/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy only necessary files for runtime
# Add only what you need - modify this as needed
COPY . .
# Add any other essential directories or files

# Expose FastAPI port
EXPOSE 80

# Set conda shell
SHELL ["conda", "run", "-n", "pano3dc", "/bin/bash", "-c"]

# Set up entrypoint
CMD ["conda", "run", "--no-capture-output", "-n", "pano3dc", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]