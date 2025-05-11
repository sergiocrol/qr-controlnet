import os
import base64
import torch
from PIL import Image
from io import BytesIO
import uuid
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel as DiffusersControlNetModel
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    DEISMultistepScheduler,
    DPMSolverSinglestepScheduler
)

DEFAULT_QR_PATH = os.path.join("/opt/program", "qrs", "qr.png")
alternative_paths = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "qrs", "qr.png"),
    "./qrs/qr.png"
]
print(f"Looking for default QR at: {DEFAULT_QR_PATH}")

RESULTS_DIR = os.environ.get('RESULTS_DIR', '/tmp/results')

os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_MAP = {
  "dreamshaper": "Lykon/DreamShaper",
  "ghostmix": "digiplay/GhostMixV1.2VAE"
}

DEFAULT_MODEL = "ghostmix"

class QRControlNetInference:
  def __init__(self, model_name=None):

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
    self.set_model(model_name)
    self.controlnet_model = os.environ.get('CONTROLNET_MODEL', "monster-labs/control_v1p_sd15_qrcode_monster")
    self.controlnet_two_model = os.environ.get('CONTROLNET_TWO_MODEL', "latentcat/control_v1p_sd15_brightness")

    self.pipe = None
    self.load_model()

  def set_model(self, model_name=None):
    """Set the base model to use for generation"""
    if model_name is None:
        model_name = os.environ.get('MODEL_NAME', DEFAULT_MODEL)
        
    model_name = model_name.lower()
    
    if model_name in MODEL_MAP:
        self.model = MODEL_MAP[model_name]
    else:
        # Default to DreamShaper if the model name is not recognized
        print(f"Warning: Model '{model_name}' not found in model map. Using {DEFAULT_MODEL} instead.")
        self.model = MODEL_MAP[DEFAULT_MODEL]
    
    print(f"Using model: {self.model}")
    return self.model
  
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

    self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
      self.pipe.scheduler.config,
      algorithm_type="dpmsolver++",
      use_karras_sigmas=True
    )

    self.pipe = self.pipe.to(self.device)
    
    print("Model loading complete")

  def get_scheduler(self, sampler_name="dpm++_2m_karras"):
    config = self.pipe.scheduler.config
    
    schedulers = {
      "ddim": (DDIMScheduler, {}),
      "pndm": (PNDMScheduler, {}),
      "lms": (LMSDiscreteScheduler, {}),
      "euler": (EulerDiscreteScheduler, {}),
      "euler_a": (EulerAncestralDiscreteScheduler, {}),
      "dpm++_2m": (DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++"}),
      "dpm++_2m_karras": (DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++", "use_karras_sigmas": True}),
      "dpm++_sde": (DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++", "solver_order": 2, "use_karras_sigmas": True}),
      "dpm++_sde_karras": (DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++", "solver_order": 2, "use_karras_sigmas": True}),
      "heun": (HeunDiscreteScheduler, {}),
      "dpm_2": (KDPM2DiscreteScheduler, {}),
      "dpm_2_a": (KDPM2AncestralDiscreteScheduler, {}),
      "unipc": (UniPCMultistepScheduler, {}),
      "deis": (DEISMultistepScheduler, {}),
      "dpm_fast": (DPMSolverSinglestepScheduler, {})
    }
    
    if sampler_name.lower() not in schedulers:
      print(f"Warning: Sampler {sampler_name} not recognized, defaulting to 'dpm++_2m_karras'")
      sampler_name = "dpm++_2m_karras"
    
    scheduler_class, params = schedulers[sampler_name.lower()]
    return scheduler_class.from_config(config, **params)

  def generate(self, prompt, **kwargs):
    print(f"Starting generation with prompt: {prompt}")
    print(f"kwargs: {kwargs}")

    seed = kwargs.get("seed", None)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        print(f"Set seed to: {seed}")

    try:
      # Load and preprocess the QR code image to match Automatic1111's behavior
      image = Image.open(DEFAULT_QR_PATH).convert("RGB")
      print(f"Loaded default QR code with size: {image.size}")
      
      # Resize image to match Processor Res: 512 from Automatic1111
      target_size = (512, 512)
      image = image.resize(target_size, Image.LANCZOS)
      print(f"Resized QR code to: {image.size}")
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
            # Apply same preprocessing
            image = image.resize(target_size, Image.LANCZOS)
            print(f"Successfully loaded and resized QR code from {alt_path}")
            break
        except Exception as alt_e:
            print(f"Failed to load from {alt_path}: {str(alt_e)}")
      else:
        raise
    
    sampler = kwargs.get("sampler", "dpm++_2m_karras")
    if sampler:
      self.pipe.scheduler = self.get_scheduler(sampler)
      print(f"Using sampler: {sampler}")

    # Configure ControlNet parameters to match Automatic1111
    controlnet_scale = kwargs.get("controlnet_conditioning_scale")
    if controlnet_scale is None:
        controlnet_scale = [1.25, 0.1]  # Default to match A1111 example
    elif not isinstance(controlnet_scale, list):
        controlnet_scale = [controlnet_scale, controlnet_scale]
    
    start = kwargs.get("control_guidance_start")
    end = kwargs.get("control_guidance_end")  

    # Final image dimensions
    height = kwargs.get("height", 1024)
    width = kwargs.get("width", 1024)
    
    # Ensure dimensions are divisible by 8
    height = (height // 8) * 8
    width = (width // 8) * 8

    params = {
        "negative_prompt": kwargs.get("negative_prompt", "ugly, disfigured, low quality, blurry, nsfw"),
        "num_inference_steps": kwargs.get("num_inference_steps", 40),  # Match A1111 default
        "controlnet_conditioning_scale": controlnet_scale,
        "control_guidance_start": [0.0, 0.1] if start is None else (start if isinstance(start, list) else [start, start]),
        "control_guidance_end": [1.0, 1.0] if end is None else (end if isinstance(end, list) else [end, end]),
        "height": height,
        "width": width,
        "guidance_scale": kwargs.get("guidance_scale", 7.0),  # CFG scale from A1111
        "eta": 0.0,  # Match A1111's default
    }

     # Add guess_mode parameter to match A1111's "Control Mode: Balanced"
    guess_mode = kwargs.get("guess_mode", False)
    
    print(f"Parameter values: {params}")

    print(f"Running inference with prompt: {prompt[:50]}...")

    # Final image processing - resize to match "Crop and Resize" mode
    # This ensures the QR code image matches the output dimensions
    qr_image_processed = image.resize((width, height), Image.LANCZOS)

    result = self.pipe(
        prompt=prompt,
        image=[qr_image_processed, qr_image_processed],  # Same image for both controlnets
        guess_mode=[guess_mode, guess_mode],  # Control Mode: Balanced
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
        "output_path": output_path,
        "seed": seed if seed else "random"
    }
  
model = None

def get_model():
    global model
    if model is None:
        model = QRControlNetInference()
    return model