import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from ..utils.logging import get_logger
import platform
import os

logger = get_logger(__name__)

def init_models(app):
    """Initialize the ControlNet models with memory optimizations"""
    logger.info("Initializing models...")
    
    device_preference = os.environ.get('DEVICE', 'cuda')
    logger.info(f"DEVICE: {device_preference}")

    if device_preference == 'cpu':
        device = "cpu"
        logger.info("Using CPU as per environment preference")
    elif device_preference == 'cuda' and torch.cuda.is_available():
        device = "cuda"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Using CUDA as per environment preference")
    else:
        if torch.cuda.is_available():
            device = "cuda"
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info(f"Using CUDA with {torch.cuda.device_count()} GPU(s)")
            logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.warning("No GPU acceleration available. Using CPU which will be much slower.")

    app.config['DEVICE'] = device
    logger.info(f"Using device: {device}")
    logger.info(f"Running on: {platform.platform()}")

    try:
        logger.info(f"Loading base model: {app.config['MODEL']}")
        logger.info(f"Loading ControlNet models: {app.config['CONTROLNET_MODEL']} and {app.config['CONTROLNET_TWO_MODEL']}")
        
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        qr_model_options = [
            app.config['CONTROLNET_MODEL'] 
        ]
        
        brightness_model_options = [
            app.config['CONTROLNET_TWO_MODEL']               
        ]
        
         # ControlNet model (qrcode monster)
        controlnet = None
        qr_model_loaded = False
        
        for model_id in qr_model_options:
            try:
                logger.info(f"Attempting to load QR ControlNet model: {model_id}")
                
                try:
                    controlnet = ControlNetModel.from_pretrained(
                        model_id,
                        subfolder="v2",
                        torch_dtype=torch_dtype,
                        use_safetensors=True
                    )
                    logger.info(f"Successfully loaded QR ControlNet model: {model_id} (with safetensors)")
                except:
                    # Fallback to .bin format
                    controlnet = ControlNetModel.from_pretrained(
                        model_id,
                        subfolder="v2",
                        torch_dtype=torch.float16,
                        use_safetensors=False
                    )
                    logger.info(f"Successfully loaded QR ControlNet model: {model_id} (with .bin format)")
                    
                qr_model_loaded = True
                break   
            except Exception as e:
                logger.warning(f"Failed to load QR model {model_id} without subfolder: {str(e)}")
                try:
                    controlnet = ControlNetModel.from_pretrained(
                        model_id,
                        subfolder="v2",
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
                try:
                    controlnet_two = ControlNetModel.from_pretrained(
                        model_id,
                        torch_dtype=torch_dtype,
                        use_safetensors=True
                    )
                    logger.info(f"Successfully loaded Brightness ControlNet model: {model_id} (with safetensors)")
                except:
                    # Fallback to .bin format
                    controlnet_two = ControlNetModel.from_pretrained(
                        model_id,
                        torch_dtype=torch_dtype,
                        use_safetensors=False
                    )
                    logger.info(f"Successfully loaded Brightness ControlNet model: {model_id} (with .bin format)")
                
                brightness_model_loaded = True
                break
            except Exception as e:
                logger.warning(f"Failed to load Brightness model {model_id}: {str(e)}")
        
        if not brightness_model_loaded:
            logger.warning("Failed to load Brightness ControlNet model. Proceeding with QR model only.")
            controlnet = [controlnet]
        else:
            logger.info("Both ControlNet models loaded successfully")
            controlnet = [controlnet, controlnet_two]
        
        logger.info(f"Loading Stable Diffusion pipeline: {app.config['MODEL']}")
        try:
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                app.config['MODEL'],
                torch_dtype=torch_dtype,
                use_safetensors=True,
                controlnet=controlnet,
                safety_checker=None,
                requires_safety_checker=False
            )
            logger.info(f"Successfully loaded pipeline: {app.config['MODEL']} (with safetensors)")
        except Exception as e:
            logger.warning(f"Failed to load pipeline with safetensors: {str(e)}")
            logger.info("Falling back to .bin format...")
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                app.config['MODEL'],
                torch_dtype=torch_dtype,
                use_safetensors=False,
                controlnet=controlnet,
                safety_checker=None,
                requires_safety_checker=False
            )
            logger.info(f"Successfully loaded pipeline: {app.config['MODEL']} (with .bin format)")

        logger.info(f"Moving models to {device} device...")
        pipe.to(device)

        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            app.config['MODEL'],
            subfolder="scheduler",
            use_karras_sigmas=True,
            thresholding=False,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            solver_order=2,
            lower_order_final=True,
        )

        pipe.scheduler = scheduler
        logger.info("Configured DPM++ 2M Karras scheduler to match Automatic1111")

        logger.info("Applying memory optimizations...")
        
        if hasattr(pipe, 'enable_vae_slicing'):
            pipe.enable_vae_slicing()
            logger.info("Enabled VAE slicing")
        
        if hasattr(pipe, 'enable_vae_tiling'):
            pipe.enable_vae_tiling()
            logger.info("Enabled VAE tiling")
        
        if hasattr(pipe, 'enable_attention_slicing'):
            pipe.enable_attention_slicing("max")
            logger.info("Enabled attention slicing with 'max' config")
        
        if device == "cuda":
            # Enable xformers if available
            try:
                pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
                logger.info("Enabled xformers memory efficient attention")
            except ImportError:
                logger.warning("xformers not available, falling back to default attention")
                
            # Enable memory efficient attention
            if hasattr(pipe, 'enable_memory_efficient_attention'):
                pipe.enable_memory_efficient_attention()
                logger.info("Enabled memory efficient attention")
            
            # Enable SDPA (Scaled Dot Product Attention) if available
            if hasattr(pipe, 'enable_sdpa'):
                pipe.enable_sdpa()
                logger.info("Enabled SDPA")

            # Enable SDPA (Scaled Dot Product Attention) if available
            # This is the default in PyTorch 2.0+
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                logger.info("PyTorch 2.0+ SDPA available by default")
        
        # Apply additional memory optimizations
        if hasattr(pipe.scheduler, 'set_timesteps'):
            # Cache timesteps for faster inference
            logger.info("Pre-computing timesteps for faster inference")
            
        if device == "cuda":
            pipe.enable_model_cpu_offload()
            logger.info("Enabled model CPU offloading")
        
        logger.info("Models initialized successfully with all optimizations")
        return pipe
        
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        raise RuntimeError(f"Failed to initialize models: {str(e)}")