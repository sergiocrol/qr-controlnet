FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

RUN apt-get update && \
  apt-get install -y python3-pip python3-dev libgl1-mesa-glx

# Install dependencies
COPY requirements.txt /opt/requirements.txt
RUN pip install -r /opt/requirements.txt

# Set up environment variables
ENV MODEL="Lykon/DreamShaper"
ENV CONTROLNET_MODEL="monster-labs/control_v1p_sd15_qrcode_monster"
ENV CONTROLNET_TWO_MODEL="latentcat/control_v1p_sd15_brightness"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /opt/program
COPY . /opt/program/
ENV PYTHONPATH=/opt/program

ENTRYPOINT ["python", "app/serve.py"]

# Expose port 8080 for SageMaker
EXPOSE 8080