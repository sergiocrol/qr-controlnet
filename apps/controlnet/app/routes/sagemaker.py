from flask import Blueprint, jsonify, request, current_app
from ..utils.logging import get_logger
from ..utils.validation import validate_schema
from ..schemas.generate import SageMakerRequestSchema
import boto3
import json
import os

logger = get_logger(__name__)
sagemaker_bp = Blueprint('sagemaker', __name__)

@sagemaker_bp.route('/generate/async', methods=['POST'])
@validate_schema(SageMakerRequestSchema)
def generate_async(data):
  try:
    prompt = data["prompt"]

    logger.info(f"Submitting async generation request with prompt: {prompt[:50]}...")

    runtime = boto3.client('sagemaker-runtime')

    payload = {
      "prompt": prompt,
      "negative_prompt": data.get("negative_prompt", "ugly, blurry, pixelated, low quality, text, watermark"),
      "num_inference_steps": data.get("num_inference_steps", current_app.config['NUM_INFERENCE_STEPS']),
      "controlnet_conditioning_scale": data.get("controlnet_conditioning_scale", [1.0, 0.5]),
      "control_guidance_start": data.get("control_guidance_start", [0.0, 0.0]),
      "control_guidance_end": data.get("control_guidance_end", [1.0, 1.0]),
      "height": data.get("height", 768),
      "width": data.get("width", 768),
    }

    response = runtime.invoke_endpoint_async(
      EndpointName="controlnet-qr-endpoint",
      ContentType="application/json",
      Accept="application/json",
      InputLocation=json.dumps(payload)
    )

    output_location = response['OutputLocation']

    logger.info(f"Successfully submitted async request. Output will be available at: {output_location}")

    return jsonify({
      "message": "Async image generation request submitted successfully",
      "request_id": response['InferenceId'],
      "output_location": output_location,
      "status": "PROCESSING"
    })
  
  except Exception as e:
    logger.exception(f"Error submitting async generation request: {str(e)}")
    return jsonify({"error": f"Async generation request failed: {str(e)}"}), 500
  
@sagemaker_bp.route('/generate/async/status/<request_id>', methods=['GET'])
def check_async_status(request_id):
  try:
    sagemaker_client = boto3.client('sagemaker')

    # Check async inference job status
    response = sagemaker_client.describe_async_inference_job(
      JobName=request_id
    )

    status = response['Status']

    if status == 'COMPLETED':
      s3_uri = response['OutputConfig']['S3OutputPath']
      bucket_name = s3_uri.split('/')[2]
      prefix = '/'.join(s3_uri.split('/')[3:])
      
      s3_client = boto3.client('s3')
      response = s3_client.get_object(
        Bucket=bucket_name,
        Key=f"{prefix}/{request_id}/output.json"
      )
      
      result = json.loads(response['Body'].read().decode())
      
      return jsonify({
        "status": status,
        "result": result
      })
    else:
      return jsonify({
        "status": status
      })
  
  except Exception as e:
    logger.exception(f"Error checking async request status: {str(e)}")
    return jsonify({"error": f"Failed to check request status: {str(e)}"}), 500
  
@sagemaker_bp.route('/generate/async/result/<request_id>', methods=['GET'])
def get_async_result(request_id):
  try:
    s3_client = boto3.client('s3')
    
    sagemaker_session = boto3.Session().client('sagemaker')
    default_bucket = sagemaker_session.describe_endpoint(EndpointName="controlnet-qr-endpoint")['EndpointArn'].split(':')[5].split('/')[0]
    
    try:
      response = s3_client.get_object(
        Bucket=default_bucket,
        Key=f"controlnet-qr-output/{request_id}/output.json"
      )
      
      result = json.loads(response['Body'].read().decode())
      
      if 'image' in result and result['image']:
        pass
          
      return jsonify({
        "status": "COMPLETED",
        "result": result
      })
        
    except s3_client.exceptions.NoSuchKey:
      return jsonify({
        "status": "PROCESSING",
        "message": "Result not available yet"
      })
          
  except Exception as e:
    logger.exception(f"Error retrieving async result: {str(e)}")
    return jsonify({"error": f"Failed to retrieve result: {str(e)}"}), 500