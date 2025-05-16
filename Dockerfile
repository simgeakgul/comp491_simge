FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install Miniconda and system dependencies
RUN apt update && apt install -y wget bzip2 libgl1 libglib2.0-0 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda clean -afy

# Add conda to path
ENV PATH="/opt/conda/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy environment definition first to leverage Docker caching
COPY environment.yaml .

# Create environment before copying full codebase (caching boost)
RUN conda env create -f environment.yaml

# Install specific pip packages (optional)
RUN conda run -n pano3dc pip install timm==1.0.15 controlnet-aux==0.0.9 --no-deps

# Set conda shell
SHELL ["conda", "run", "-n", "pano3dc", "/bin/bash", "-c"]

# Copy full project files
COPY . .

# Expose FastAPI port
EXPOSE 80

# Production CMD (consider removing --reload in production)
CMD ["conda", "run", "--no-capture-output", "-n", "pano3dc", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]