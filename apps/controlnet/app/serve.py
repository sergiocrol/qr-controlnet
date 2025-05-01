import os
import json
import logging
import traceback
from flask import Flask, request, Response
import inference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a Flask app for SageMaker
app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """
    SageMaker health check endpoint.
    Must return 200 status code for the endpoint to be considered healthy.
    """
    try:
        # Try to load model to confirm it's ready
        inference.get_model()
        return Response(
            response=json.dumps({"status": "healthy"}),
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        # SageMaker will mark the endpoint as unhealthy if we return non-200
        return Response(
            response=json.dumps({"status": "unhealthy", "error": str(e)}),
            status=500,
            mimetype='application/json'
        )

@app.route('/invocations', methods=['POST'])
def invocations():
    """
    SageMaker inference endpoint.
    Handles incoming requests and returns generated images.
    """
    try:
        # Check content type
        if request.content_type != 'application/json':
            return Response(
                response=json.dumps({"error": "This predictor only supports JSON data"}),
                status=415,
                mimetype='application/json'
            )
        
        # Parse input
        input_data = request.get_json()
        logger.info(f"Received request with prompt: {input_data.get('prompt', '')[:50]}...")
        
        # Validate required parameters
        prompt = input_data.get("prompt")
        if not prompt:
            return Response(
                response=json.dumps({"error": "Missing required parameter: prompt"}),
                status=400,
                mimetype='application/json'
            )
        
        # Get model and generate image
        model = inference.get_model()
        result = model.generate(
            prompt=prompt,
            negative_prompt=input_data.get("negative_prompt"),
            num_inference_steps=input_data.get("num_inference_steps"),
            controlnet_conditioning_scale=input_data.get("controlnet_conditioning_scale"),
            control_guidance_start=input_data.get("control_guidance_start"),
            control_guidance_end=input_data.get("control_guidance_end"),
            height=input_data.get("height"),
            width=input_data.get("width"),
            guidance_scale=input_data.get("guidance_scale")
        )
        
        # Return result
        return Response(
            response=json.dumps(result),
            status=200,
            mimetype='application/json'
        )
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error during inference: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        
        return Response(
            response=json.dumps({
                "error": str(e),
                "traceback": error_traceback
            }),
            status=500,
            mimetype='application/json'
        )

# For local development, run the app if this script is executed directly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)