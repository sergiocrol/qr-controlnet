#!/usr/bin/env python
"""
Test client for the SageMaker local container.
This script sends a sample request to the SageMaker container running locally
and saves the output image.
"""

import json
import requests
import base64
import argparse
import os
from PIL import Image
from io import BytesIO
import time

def test_sagemaker_endpoint(prompt, steps=20, endpoint_url="http://localhost:8080"):
    """
    Send a test request to the SageMaker endpoint.
    
    Args:
        prompt (str): The text prompt for image generation
        steps (int): Number of inference steps
        endpoint_url (str): URL of the SageMaker endpoint
        
    Returns:
        bool: True if successful, False otherwise
    """
    # First check if the endpoint is healthy
    try:
        response = requests.get(f"{endpoint_url}/ping")
        if response.status_code != 200:
            print(f"Endpoint health check failed with status {response.status_code}: {response.text}")
            return False
        print("Endpoint is healthy!")
    except Exception as e:
        print(f"Failed to connect to endpoint: {str(e)}")
        return False
    
    # Prepare the payload
    payload = {
        "prompt": prompt,
        "negative_prompt": "ugly, disfigured, low quality, blurry, nsfw",
        "num_inference_steps": steps,
        "controlnet_conditioning_scale": [1.25, 0.1],
        "control_guidance_start": [0, 0.1],
        "control_guidance_end": [1, 1],
        "height": 512,
        "width": 512
    }
    
    try:
        # Start the timer
        start_time = time.time()
        
        # Send the request
        print(f"Sending request to {endpoint_url}/invocations with prompt: '{prompt}'")
        response = requests.post(
            f"{endpoint_url}/invocations",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        # Check if successful
        if response.status_code != 200:
            print(f"Request failed with status {response.status_code}: {response.text}")
            return False
        
        # Calculate time taken
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Request completed in {elapsed_time:.2f} seconds")
        
        # Parse the response
        result = response.json()
        
        # Check if image exists in response
        if "image" not in result:
            print("No image found in response.")
            return False
        
        # Save the image
        img_data = base64.b64decode(result["image"])
        img = Image.open(BytesIO(img_data))
        
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Save the image
        output_path = f"results/sagemaker_test_{int(time.time())}.png"
        img.save(output_path)
        print(f"Image saved to {output_path}")
        
        return True
    
    except Exception as e:
        print(f"Error during test: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SageMaker local container")
    parser.add_argument("--prompt", type=str, default="A QR code in the style of Van Gogh's Starry Night",
                        help="The text prompt for image generation")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of inference steps (lower for faster testing)")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8080",
                        help="URL of the SageMaker endpoint")
    
    args = parser.parse_args()
    
    success = test_sagemaker_endpoint(args.prompt, args.steps, args.endpoint)
    
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed.")
        exit(1)