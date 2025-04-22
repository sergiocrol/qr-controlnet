import requests
import base64
import json
import argparse
import os
import time
from PIL import Image
from io import BytesIO

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')

def save_image(base64_string, output_path):
  image_data = base64.b64decode(base64_string)
  image = Image.open(BytesIO(image_data))
  image.save(output_path)
  print(f"Image saved to {output_path}")

def main():
  parser = argparse.ArgumentParser(description='Test ControlNet API')
  parser.add_argument('--url', default='http://localhost:8080/generate', help='API URL')
  parser.add_argument('--image', required=True, help='Path to input QR code image')
  parser.add_argument('--prompt', required=True, help='Generation prompt')
  parser.add_argument('--steps', type=int, default=20, help='Number of inference steps')
  parser.add_argument('--size', type=int, default=512, help='Image size (width & height)')
  parser.add_argument('--output', default='./generated_qr.png', help='Output image path')
  parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps'], help='Device to use')
  
  args = parser.parse_args()
  
  print(f"Reading image from {args.image}")
  base64_image = encode_image(args.image)
  
  payload = {
      "prompt": args.prompt,
      "input_image": base64_image,
      "negative_prompt": "ugly, disfigured, low quality, blurry, nsfw",
      "num_inference_steps": args.steps,
      "controlnet_conditioning_scale": [0.8, 0.1],
      "control_guidance_start": [0, 0.1], 
      "control_guidance_end": [1, 1],
      "height": args.size,
      "width": args.size,
      "device": args.device
  }
  
  print(f"Sending request to {args.url}")
  print(f"Prompt: {args.prompt}")
  print(f"Steps: {args.steps}")
  print(f"Size: {args.size}x{args.size}")
  
  start_time = time.time()
  response = requests.post(
      args.url, 
      json=payload,
      headers={'Content-Type': 'application/json'}
  )
  
  if response.status_code == 200:
      data = response.json()
      processing_time = time.time() - start_time
      
      print(f"Request successful in {processing_time:.2f} seconds")
      print(f"Server processing time: {data.get('processing_time_seconds', 'N/A')} seconds")
      
      save_image(data['image'], args.output)
  else:
      print(f"Error: {response.status_code}")
      print(f"Response: {response.text}")

if __name__ == "__main__":
    main()