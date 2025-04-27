import argparse
import requests
import json
import base64
from PIL import Image
import io
import time

def main():
    parser = argparse.ArgumentParser(description='Test the SageMaker container locally')
    parser.add_argument('--prompt', type=str, default='A futuristic city with neon lights',
                       help='Prompt for image generation')
    parser.add_argument('--steps', type=int, default=10,
                       help='Number of inference steps')
    parser.add_argument('--port', type=int, default=8081,
                       help='Port where the SageMaker container is running')
    
    args = parser.parse_args()
    
    # Check if the container is running
    try:
        ping_response = requests.get(f"http://localhost:{args.port}/ping")
        if ping_response.status_code != 200:
            print(f"Error: SageMaker container is not responding. Status code: {ping_response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to SageMaker container at http://localhost:{args.port}")
        print("Make sure the container is running with 'make sagemaker-run'")
        return
        
    print(f"SageMaker container is responsive. Status: {ping_response.text}")
    
    # Prepare the payload
    payload = {
        "prompt": args.prompt,
        "num_inference_steps": args.steps,
        "negative_prompt": "ugly, blurry, pixelated, low quality, text, watermark"
    }
    
    print(f"Sending request with prompt: '{args.prompt}'")
    print(f"This will take some time as it's performing inference with {args.steps} steps...")
    
    start_time = time.time()
    
    # Send the request
    try:
        response = requests.post(
            f"http://localhost:{args.port}/invocations",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return
    
    end_time = time.time()
    duration = end_time - start_time
    
    if response.status_code != 200:
        print(f"Error: Request failed with status {response.status_code}")
        print(f"Response: {response.text}")
        return
    
    # Parse the response
    try:
        result = response.json()
    except json.JSONDecodeError:
        print("Error: Couldn't parse JSON response")
        print(f"Raw response: {response.text[:100]}...")
        return
    
    print(f"Request succeeded in {duration:.2f} seconds!")
    
    # If there's an image in the response, save it
    if "image" in result:
        try:
            image_data = base64.b64decode(result["image"])
            image = Image.open(io.BytesIO(image_data))
            output_filename = f"sagemaker_test_output_{int(time.time())}.png"
            image.save(output_filename)
            print(f"Generated image saved as {output_filename}")
        except Exception as e:
            print(f"Error saving image: {e}")
    
    # Print output path if available
    if "output_path" in result:
        print(f"Output path: {result['output_path']}")

if __name__ == "__main__":
    main()