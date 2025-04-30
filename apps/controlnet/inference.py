import os
import base64
import torch
from PIL import Image
from io import BytesIO
import uuid
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel as DiffusersControlNetModel

DEFAULT_QR_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "qrs", "qr.png")
print(f"Looking for default QR at: {DEFAULT_QR_PATH}")

RESULTS_DIR = os.environ.get('RESULTS_DIR', '/tmp/results')

os.makedirs(RESULTS_DIR, exist_ok=True)

class QRControlNetInference:
  def __init__(self):

    import subprocess
    print("GPU information:")

    try:
      subprocess.run(["nvidia-smi"], check=False)
    except:
      print("Failed to run nvidia-smi") 

    # First set the device
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        self.device = "cuda"
    else:
        self.device = "cpu"
    
    print(f"Using device: {self.device}")

    # Model configs
    self.model = os.environ.get('MODEL', "Lykon/DreamShaper")
    self.controlnet_model = os.environ.get('CONTROLNET_MODEL', "monster-labs/control_v1p_sd15_qrcode_monster")
    self.controlnet_two_model = os.environ.get('CONTROLNET_TWO_MODEL', "latentcat/control_v1p_sd15_brightness")

    self.pipe = None
    self.load_model()
  
  def load_model(self):
    print("Loading controlnet models...")

    controlnet = DiffusersControlNetModel.from_pretrained(
      self.controlnet_model,
      torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
    )

    controlnet_two = DiffusersControlNetModel.from_pretrained(
      self.controlnet_two_model,
      torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
    )

    print("Loading stable diffusion pipeline...")
    self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
      self.model,
      controlnet=[controlnet, controlnet_two],
      torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
    )

    self.pipe = self.pipe.to(self.device)

    # model memory optimizations (OPTIONAL. Remove if not working properly)
    if self.device == "cuda":
      self.pipe.enable_xformers_memory_efficient_attention()
    
    print("Model loading complete")

  def generate(self, prompt, **kwargs):
    """Generate an image based on a prompt and a QR code"""
    print(f"Starting generation with prompt: {prompt}")
    print(f"kwargs: {kwargs}")

    try:
      image = Image.open(DEFAULT_QR_PATH).convert("RGB")
      print(f"Loaded default QR code with size: {image.size}")
    except Exception as e:
      print(f"Failed to load default QR code from {DEFAULT_QR_PATH}: {str(e)}")
      alternative_paths = [
        "/opt/program/qrs/qr.png",
        "/opt/qrs/qr.png",
        "/qrs/qr.png",
        "./qrs/qr.png"
      ]
      for alt_path in alternative_paths:
        try:
          print(f"Trying alternative path: {alt_path}")
          image = Image.open(alt_path).convert("RGB")
          print(f"Successfully loaded QR code from {alt_path}")
          break
        except Exception as alt_e:
          print(f"Failed to load from {alt_path}: {str(alt_e)}")
      else:
        raise

    controlnet_scale = kwargs.get("controlnet_conditioning_scale")
    if controlnet_scale is None:
      controlnet_scale = [1.0, 0.5]
    elif not isinstance(controlnet_scale, list):
      controlnet_scale = [controlnet_scale, controlnet_scale]
    
    start = kwargs.get("control_guidance_start")
    end = kwargs.get("control_guidance_end")  

    params = {
      "negative_prompt": kwargs.get("negative_prompt", "ugly, blurry, pixelated, low quality, text, watermark"),
      "num_inference_steps": kwargs.get("num_inference_steps", 30),
      "controlnet_conditioning_scale": controlnet_scale,
      "control_guidance_start": [0.0, 0.0] if start is None else (start if isinstance(start, list) else [start, start]),
      "control_guidance_end": [1.0, 1.0] if end is None else (end if isinstance(end, list) else [end, end]),
      "height": kwargs.get("height", 768),
      "width": kwargs.get("width", 768),
      "guidance_scale": kwargs.get("guidance_scale", 7.5),
    }
    print(f"Parameter values: {params}")

    print(f"Running inference with prompt: {prompt[:50]}...")

    result = self.pipe(
      prompt=prompt,
      image=[image, image],  # Same image for both controlnets
      **params
    )

    # Get the generated image
    generated_image = result.images[0]

    # Save the image
    unique_id = str(uuid.uuid4())[:8]
    output_path = os.path.join(RESULTS_DIR, f'output_{unique_id}.png')
    generated_image.save(output_path)
    print(f"Saved generated image to {output_path}")

    # Convert the output image to base64 for response
    buffered = BytesIO()
    generated_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {
      "image": img_str,
      "output_path": output_path
    }
  
model = None

def get_model():
    global model
    if model is None:
        model = QRControlNetInference()
    return model