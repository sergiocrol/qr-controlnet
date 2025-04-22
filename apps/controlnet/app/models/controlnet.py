import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from ..utils.logging import get_logger
import platform
import os
import gc

logger = get_logger(__name__)

def init_models(app):
    """Initialize the ControlNet models with memory optimizations"""
    logger.info("Initializing models...")
    
    device_preference = os.environ.get('PREFERRED_DEVICE', 'auto')
    
    if device_preference == 'cpu':
        device = "cpu"
        logger.info("Using CPU as per environment preference")
    elif device_preference == 'mps' and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        logger.info("Using MPS (Apple Silicon) as per environment preference")
    elif device_preference == 'cuda' and torch.cuda.is_available():
        device = "cuda"
        logger.info("Using CUDA as per environment preference")
    else:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
            logger.info("Using MPS (Apple Silicon) acceleration")
        elif torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA with {torch.cuda.device_count()} GPU(s)")
            logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.warning("No GPU acceleration available. Using CPU which will be much slower.")

    app.config['DEVICE'] = device
    logger.info(f"Using device: {device}")
    logger.info(f"Running on: {platform.platform()}")

    try:
        if device == "mps":
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
            logger.info(f"Set MPS high watermark ratio to {os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']}")

        gc.collect()
        
        logger.info(f"Loading base model: {app.config['MODEL']}")
        logger.info(f"Loading ControlNet models: {app.config['CONTROLNET_MODEL']} and {app.config['CONTROLNET_TWO_MODEL']}")
        
        logger.info("Loading ControlNet models with memory optimization...")
        
        qr_model_options = [
            app.config['CONTROLNET_MODEL'],                    
            "monster-labs/control_v1p_sd15_qrcode_monster",   
            "monster-labs/control_v1p_sd15_qrcode",            
            "monster-labs/control_v1p_sd15_qrcode_monster_v2"  
        ]
        
        brightness_model_options = [
            app.config['CONTROLNET_TWO_MODEL'],                
            "latentcat/control_v1p_sd15_brightness",           
            "lllyasviel/control_v11p_sd15_brightness",         
            "lllyasviel/sd-controlnet-brightness",             
            "SG161222/Realistic_Vision_V5.1_noVAE",            
            "furusu/control_v1p_sd15_brightness_fp16",         
            "ioclab/control_v1p_sd15_brightness"               
        ]
        
         # ControlNet model (qrcode monster)
        controlnet = None
        qr_model_loaded = False
        
        for model_id in qr_model_options:
            try:
                logger.info(f"Attempting to load QR ControlNet model: {model_id}")
                
                controlnet = ControlNetModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True
                )
                logger.info(f"Successfully loaded QR ControlNet model: {model_id}")
                qr_model_loaded = True
                break
            except Exception as e:
                logger.warning(f"Failed to load QR model {model_id} without subfolder: {str(e)}")
                try:
                    # Try with subfolder v2
                    controlnet = ControlNetModel.from_pretrained(
                        model_id,
                        subfolder="v2",
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                        low_cpu_mem_usage=True
                    )
                    logger.info(f"Successfully loaded QR ControlNet model: {model_id} (with subfolder v2)")
                    qr_model_loaded = True
                    break
                except Exception as e2:
                    logger.warning(f"Failed to load QR model {model_id} with subfolder v2: {str(e2)}")
        
        if not qr_model_loaded:
            raise RuntimeError("Failed to load any QR ControlNet model. Please check model availability.")
            
        # ControlNet model (Brightness)
        controlnet_two = None
        brightness_model_loaded = False
        
        for model_id in brightness_model_options:
            try:
                logger.info(f"Attempting to load Brightness ControlNet model: {model_id}")
                controlnet_two = ControlNetModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True
                )
                logger.info(f"Successfully loaded Brightness ControlNet model: {model_id}")
                brightness_model_loaded = True
                break
            except Exception as e:
                logger.warning(f"Failed to load Brightness model {model_id}: {str(e)}")
        
        if not brightness_model_loaded:
            logger.warning("Failed to load Brightness ControlNet model. Proceeding with QR model only.")
            controlnet = [controlnet]
        else:
            controlnet = [controlnet, controlnet_two]
        
        logger.info(f"Loading Stable Diffusion pipeline: {app.config['MODEL']}")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            app.config['MODEL'],
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )

        logger.info(f"Moving models to {device} device...")
        pipe.to(device)

        logger.info("Applying memory optimizations...")
        
        if hasattr(pipe, 'enable_attention_slicing'):
            pipe.enable_attention_slicing(1)
            logger.info("Enabled attention slicing for memory optimization")
        
        if hasattr(pipe, 'enable_vae_slicing'):
            pipe.enable_vae_slicing()
            logger.info("Enabled VAE slicing")
        
        if device == "cuda":
            try:
                pipe.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention")
            except Exception as e:
                logger.warning(f"Could not enable xformers: {str(e)}")
        
        logger.info("Models initialized successfully")
        return pipe
        
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        raise RuntimeError(f"Failed to initialize models: {str(e)}")