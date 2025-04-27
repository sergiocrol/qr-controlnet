# app/routes/generate.py
from flask import Blueprint, jsonify, request, current_app
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import uuid
import os
import time
import torch
from ..utils.logging import get_logger
from ..utils.validation import validate_schema
from ..schemas.generate import GenerateImageSchema

logger = get_logger(__name__)
generate_bp = Blueprint('generate', __name__)

@generate_bp.route('/generate', methods=['POST'])
@validate_schema(GenerateImageSchema)
def generate_image(data):
    start_time = time.time()
    
    try:
        prompt = data["prompt"]
        image_input = data["input_image"]
        negative_prompt = data["negative_prompt"]
        num_inference_steps = data.get("num_inference_steps", current_app.config['NUM_INFERENCE_STEPS'])
        controlnet_conditioning_scale = data["controlnet_conditioning_scale"]
        control_guidance_start = data["control_guidance_start"]
        control_guidance_end = data["control_guidance_end"]
        height = data["height"]
        width = data["width"]
        device_override = data.get("device", None)
        
        if torch.cuda.is_available():
            logger.info(f"CUDA memory before load: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        logger.info(f"Processing generation request with prompt: {prompt[:50]}...")
        
        use_cpu = False
        if device_override == "cpu":
            logger.info("Using CPU for this request as requested")
            use_cpu = True
        
        try:
            image_data = base64.b64decode(image_input)
            image = Image.open(BytesIO(image_data)).convert("RGB")
            logger.debug(f"Loaded input image with size: {image.size}")
            
            # Resize image if it's too large
            if image.width > 768 or image.height > 768:
                logger.info(f"Resizing large input image from {image.size} to 768px max")
                image.thumbnail((768, 768), Image.LANCZOS)
        except Exception as e:
            logger.error(f"Failed to load input image: {str(e)}")
            return jsonify({"error": f"Failed to load input image: {str(e)}"}), 400

        pipe = current_app.pipe
        if pipe is None:
            logger.error("Model pipeline not initialized")
            return jsonify({"error": "Model not initialized"}), 500
        
        original_device = current_app.config['PREFERRED_DEVICE']
        if use_cpu and original_device != "cpu":
            logger.info("Moving model to CPU for this request")
            try:
                pipe.to("cpu")
            except Exception as e:
                logger.error(f"Failed to move model to CPU: {str(e)}")
                use_cpu = False
        
        # Apply memory optimizations for MPS (mac M1)
        if current_app.config['PREFERRED_DEVICE'] == "mps" and not use_cpu:
            logger.info("Applying aggressive memory optimizations for MPS")
            
            import gc
            gc.collect()
            
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
                logger.info("Cleared MPS cache")
        
        device_for_inference = "cpu" if use_cpu else current_app.config['PREFERRED_DEVICE']
        logger.info(f"Running inference with {num_inference_steps} steps on {device_for_inference}...")
        
        
        with torch.inference_mode():
            try:
                try:
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=[image, image],
                        num_inference_steps=num_inference_steps,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        control_guidance_start=control_guidance_start,
                        control_guidance_end=control_guidance_end,
                        height=height,
                        width=width,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and device_for_inference != "cpu":
                        logger.warning(f"OOM error: {str(e)}")
                        logger.info("Retrying with CPU due to memory constraints")
                        
                        pipe.to("cpu")
                        
                        result = pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image=[image, image],
                            num_inference_steps=min(num_inference_steps, 20),
                            controlnet_conditioning_scale=controlnet_conditioning_scale,
                            control_guidance_start=control_guidance_start,
                            control_guidance_end=control_guidance_end,
                            height=min(height, 512),
                            width=min(width, 512),
                        )
                        
                        if original_device != "cpu":
                            try:
                                pipe.to(original_device)
                            except Exception as move_error:
                                logger.error(f"Error moving back to original device: {str(move_error)}")
                    else:
                        raise
                
                generated_image = result.images[0]
            except Exception as gen_error:
                logger.error(f"Error during generation: {str(gen_error)}")
                
                if "out of memory" in str(gen_error).lower():
                    return jsonify({
                        "error": "Out of memory error. Try reducing image size to 512x512 or fewer inference steps (15-20).",
                        "details": str(gen_error)
                    }), 500
                raise
        
        image_array = np.array(generated_image)
        if np.isnan(image_array).any() or np.isinf(image_array).any():
            logger.error("The generated image contains NaN or infinite values")
            return jsonify({"error": "The generated image contains invalid values"}), 500

        os.makedirs(current_app.config['RESULTS_DIR'], exist_ok=True)
        unique_id = str(uuid.uuid4())[:8]
        output_path = os.path.join(current_app.config['RESULTS_DIR'], f'output_{unique_id}.png')
        generated_image.save(output_path)
        logger.info(f"Saved generated image to {output_path}")
        
        buffered = BytesIO()
        generated_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        processing_time = time.time() - start_time
        logger.info(f"Request completed in {processing_time:.2f} seconds")
        
        return jsonify({
            "message": "Image generated successfully",
            "output_path": output_path,
            "image": img_str,
            "processing_time_seconds": round(processing_time, 2),
            "device_used": device_for_inference
        })

    except Exception as e:
        logger.exception(f"Error during image generation: {str(e)}")
        return jsonify({"error": f"Image generation failed: {str(e)}"}), 500