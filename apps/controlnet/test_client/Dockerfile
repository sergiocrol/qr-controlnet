FROM python:3.10-slim

WORKDIR /app

# Install required packages
RUN pip install --no-cache-dir requests pillow

# Copy the test script
COPY test_sagemaker_local.py /app/

# Make the script executable
RUN chmod +x /app/test_sagemaker_local.py

# Create the results directory
RUN mkdir -p /app/results

# Set the entry point to the script
ENTRYPOINT ["python", "test_sagemaker_local.py"]

# Default command (can be overridden)
CMD ["--prompt", "A QR code in cyberpunk style", "--steps", "10"]