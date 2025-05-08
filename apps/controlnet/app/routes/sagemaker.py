from flask import Blueprint, jsonify, current_app
from ..utils.logging import get_logger
from ..utils.validation import validate_schema
from ..schemas.generate import SageMakerRequestSchema
import boto3
import json

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
      "controlnet_conditioning_scale": data.get("controlnet_conditioning_scale", [1.25, 0.1]),
      "control_guidance_start": data.get("control_guidance_start", [0.0, 0.01]),
      "control_guidance_end": data.get("control_guidance_end", [1.0, 1.0]),
      "height": data.get("height", 1024),
      "width": data.get("width", 1024),
      "sampler": data.get("sampler", "dpm++_2m_karras"),
      "guidance_scale": data.get("guidance_scale", 7.5),
      "model": data.get("model", "ghostmix"),
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
        
        # Get the SageMaker default bucket
        sagemaker_session = boto3.Session().client('sagemaker')
        endpoint_response = sagemaker_session.describe_endpoint(EndpointName="controlnet-qr-endpoint")
        default_bucket = endpoint_response['EndpointArn'].split(':')[5].split('/')[0]
        
        key = f"controlnet-qr-output/{request_id}/output.json"
        logger.info(f"Looking for result in s3://{default_bucket}/{key}")
        
        try:
            # Try to get the result from S3
            response = s3_client.get_object(
                Bucket=default_bucket,
                Key=key
            )
            
            # Parse the JSON result
            result = json.loads(response['Body'].read().decode())
            
            # Handle the image data - typically this is base64 encoded
            if 'image' in result and result['image']:
                # Optionally decode or process the base64 image here if needed
                # For example, you might want to verify it's valid base64
                import base64
                try:
                    base64.b64decode(result['image'])
                    logger.info("Successfully validated base64 image data")
                except Exception as img_error:
                    logger.error(f"Invalid base64 image data: {str(img_error)}")
                    result['image_valid'] = False
                else:
                    result['image_valid'] = True
                
                # Add additional metadata about the result
                result['image_size_bytes'] = len(result['image'])
                
                # Add S3 reference for direct access if needed
                result['s3_reference'] = {
                    'bucket': default_bucket,
                    'key': key
                }
                
                # If there's an output path in the result, verify it exists
                if 'output_path' in result:
                    # Extract just the filename from the path
                    filename = os.path.basename(result['output_path'])
                    # Construct the S3 key for the image file
                    image_key = f"controlnet-qr-output/{request_id}/{filename}"
                    
                    try:
                        # Check if the image file exists in S3
                        s3_client.head_object(Bucket=default_bucket, Key=image_key)
                        result['image_file_s3'] = f"s3://{default_bucket}/{image_key}"
                    except Exception as s3_err:
                        logger.warning(f"Image file not found in S3: {str(s3_err)}")
                        result['image_file_s3'] = None
            
            return jsonify({
                "status": "COMPLETED",
                "result": result,
                "request_id": request_id,
                "timestamp": int(time.time())
            })
                
        except s3_client.exceptions.NoSuchKey:
            # The key doesn't exist yet - the processing might still be ongoing
            logger.info(f"Result not yet available for request_id: {request_id}")
            
            # Check the job status
            try:
                job_response = sagemaker_session.describe_inference_job(
                    JobName=request_id
                )
                status = job_response.get('Status', 'UNKNOWN')
            except Exception as job_err:
                logger.warning(f"Could not check job status: {str(job_err)}")
                status = "PROCESSING"  # Assume still processing if we can't check
            
            return jsonify({
                "status": status,
                "message": "Result not available yet, processing in progress",
                "request_id": request_id
            })
        
        except Exception as fetch_error:
            logger.error(f"Error fetching result from S3: {str(fetch_error)}")
            return jsonify({
                "status": "ERROR",
                "error": f"Failed to fetch result: {str(fetch_error)}",
                "request_id": request_id
            }), 500
          
    except Exception as e:
        logger.exception(f"Error retrieving async result: {str(e)}")
        return jsonify({
            "error": f"Failed to retrieve result: {str(e)}",
            "request_id": request_id
        }), 500