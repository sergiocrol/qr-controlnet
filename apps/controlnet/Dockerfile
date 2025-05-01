# Use the PyTorch CUDA base image to ensure GPU compatibility
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Label the image
LABEL maintainer="Your Name <your.email@example.com>"
LABEL description="ControlNet QR Code Generator for SageMaker"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up directories expected by SageMaker
RUN mkdir -p /opt/ml/model \
    /opt/ml/input/data \
    /opt/ml/model/code \
    /opt/ml/output \
    /opt/program \
    /opt/ml/model/results

# Set the working directory
WORKDIR /opt/program

# Copy requirements and install dependencies
COPY requirements.txt /opt/program/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the QR code templates
COPY /app/qrs /opt/program/qrs

# Copy the application code
COPY . /opt/program/

# Create a symlink to ensure code is accessible via both paths
# This makes the codebase work both in SageMaker and local development
RUN ln -s /opt/program /opt/ml/model/code

# Make the serve script executable
RUN chmod +x /opt/program/serve

# Port used by the SageMaker container
EXPOSE 8080

# Environment variables
ENV MODEL="Lykon/DreamShaper" \
    CONTROLNET_MODEL="monster-labs/control_v1p_sd15_qrcode_monster" \
    CONTROLNET_TWO_MODEL="latentcat/control_v1p_sd15_brightness" \
    RESULTS_DIR="/opt/ml/model/results" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Command to run when container starts - this uses the same entry point
# whether running locally or in SageMaker
ENTRYPOINT ["python", "-m", "serve"]