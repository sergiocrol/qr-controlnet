FROM --platform=linux/arm64 pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /opt/program

# Install dependencies
COPY sagemaker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY app/models /opt/program/app/models
COPY sagemaker/inference.py /opt/program/
COPY sagemaker/serve.py /opt/program/
COPY qrs /opt/program/qrs

# Set up environment variables
ENV MODEL="Lykon/DreamShaper"
ENV CONTROLNET_MODEL="monster-labs/control_v1p_sd15_qrcode_monster"
ENV CONTROLNET_TWO_MODEL="latentcat/control_v1p_sd15_brightness"
ENV RESULTS_DIR="/tmp/results"

# Create the results directory
RUN mkdir -p /tmp/results

# Expose port 8080 for SageMaker
EXPOSE 8080

# Set entry point
ENTRYPOINT ["python", "serve.py"]