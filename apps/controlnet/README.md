# Controlnet App

This app provides an API for generating artistic QR codes using ControlNet models while maintaining the QR code's functionality.

## Features

- Transform ordinary QR codes into artistic images based on text prompts
- Maintain QR code functionality in the transformed images
- Cross-platform support (Mac, Windows, Linux)
- Optimized for Apple Silicon (M1/M2) Macs

## Getting Started

### Prerequisites

- Python 3.10+
- pip or pipenv
- For GPU acceleration:
  - Apple Silicon Mac: Native MPS support
  - NVIDIA GPU: CUDA 11.7+

### Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the server:

```bash
python server.py
```

Or using the Turborepo scripts:

```bash
pnpm dev
```

### Docker

Build and run with Docker:

```bash
# Build the image
docker build -t controlnet .

# Run the container
docker run -p 8080:8080 -v ./results:/app/results controlnet
```

Or using the package.json scripts:

```bash
pnpm docker:build
pnpm docker:run
```

## API Endpoints

### Generate QR Code

**Endpoint**: `POST /generate`

**Request Body**:

```json
{
  "prompt": "A QR code in the style of Van Gogh's Starry Night",
  "input_image": "BASE64_ENCODED_QR_CODE",
  "negative_prompt": "ugly, disfigured, low quality, blurry, nsfw",
  "num_inference_steps": 30,
  "controlnet_conditioning_scale": [1.25, 0.1],
  "control_guidance_start": [0, 0.1],
  "control_guidance_end": [1, 1],
  "height": 768,
  "width": 768
}
```

**Response**:

```json
{
  "message": "Image generated successfully",
  "output_path": "./results/output_a1b2c3d4.png",
  "image": "BASE64_ENCODED_GENERATED_IMAGE",
  "processing_time_seconds": 12.34
}
```

### Health Check

**Endpoint**: `GET /health`

**Response**:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "mps",
  "cuda_available": false,
  "mps_available": true,
  "memory_info": {},
  "system_info": {
    "platform": "macOS-13.1-arm64-arm-64bit",
    "python_version": "3.10.9",
    "pytorch_version": "2.3.1"
  },
  "results_directory": {
    "path": "./results",
    "exists": true
  }
}
```

## Models Used

- Base Model: Lykon/DreamShaper
- ControlNet QR Code: monster-labs/control_v1p_sd15_qrcode_monster
- ControlNet Brightness: latentcat/control_v1p_sd15_brightness

## Performance Tips

### For Apple Silicon (M1/M2) Macs:

- Use the default settings which are optimized for M1/M2 performance
- The first run will download the models and might take time
- Subsequent runs will be faster as models are cached

### For NVIDIA GPUs:

- Set higher inference steps (40-50) for better quality
- Increase resolution to 1024x1024 if GPU memory allows
