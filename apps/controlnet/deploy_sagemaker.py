import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ENV = "production"

load_dotenv()

env_file = BASE_DIR / ".env"
if env_file.exists():
    load_dotenv(env_file)

prod_env = BASE_DIR / ".env.production" 
if prod_env.exists():
    load_dotenv(prod_env)

import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.serverless import ServerlessInferenceConfig


def deploy_sagemaker_endpoint():
  aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
  aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
  aws_region = os.environ.get('AWS_REGION')

  if aws_access_key and aws_secret_key and aws_region:
    print(f"Using provided AWS credentials for region {aws_region}")
    boto_session = boto3.Session(
      aws_access_key_id=aws_access_key,
      aws_secret_access_key=aws_secret_key,
      region_name=aws_region
    )
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
  else:
    print("Using default AWS credentials from environment or credentials file")
    sagemaker_session = sagemaker.Session()

    if not aws_region:
      aws_region = boto_session.region_name
      print(f"Using region: {aws_region}")

  role = os.environ.get("SAGEMAKER_ROLE_ARN")
  if not role:
    try:
      print("No SAGEMAKER_ROLE_ARN provided, attempting to find a suitable role...")
      iam = boto3.client('iam')
      roles = iam.list_roles()
      for r in roles['Roles']:
        if 'AmazonSageMaker' in r['RoleName']:
          role = r['Arn']
          print(f"Found SageMaker role: {role}")
          break
    except Exception as e:
      print(f"Warning: Could not automatically find a SageMaker role: {e}")
  else: 
    print(f"Using provided SageMaker role: {role}")

  if not role:
    raise ValueError("Could not find SageMaker execution role. Please provide it as SAGEMAKER_ROLE_ARN environment variable.")
  
  # Get S3 bucket
  s3_bucket = os.environ.get('AWS_S3_BUCKET')
  if s3_bucket:
    bucket = s3_bucket
    print(f"Using specified S3 bucket: {bucket}")
  else:
    bucket = sagemaker_session.default_bucket()
    print(f"Using default bucket: {bucket}")

  # Ensure bucket exists
  s3 = boto_session.client('s3')
  try:
    s3.head_bucket(Bucket=bucket)
  except:
    print(f"Bucket {bucket} does not exist, creating it...")
    if aws_region == 'us-east-1':
      s3.create_bucket(Bucket=bucket)
    else:
      s3.create_bucket(
        Bucket=bucket,
        CreateBucketConfiguration={'LocationConstraint': aws_region}
      )
  
  repository_name = os.environ.get("AWS_ECR_REPOSITORY_NAME", "controlnet-ai-qr-generator")

  # Build + push docker image
  print(f"Building and pushing Docker image to ECR repository: {repository_name}")
  image_uri = sagemaker_session.push_to_ecr(
    repository_name=repository_name,
    image_path="./sagemaker"
  )
  print(f"Image URI: {image_uri}")

  # Configure serverless inference
  memory_size = int(os.environ.get("SAGEMAKER_MEMORY_SIZE_MB", "4096"))
  max_concurrency = int(os.environ.get("SAGEMAKER_MAX_CONCURRENCY", "2"))

  print(f"Configuring serverless inference with memory: {memory_size}MB, concurrency: {max_concurrency}")
  serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=memory_size, 
    max_concurrency=max_concurrency
  )

   # Create the model
  model_name = os.environ.get("SAGEMAKER_MODEL_NAME", "controlnet-qr-model")
  print(f"Creating SageMaker model: {model_name}")

  model = Model(
    image_uri=image_uri,
    role=role,
    name=model_name,
    sagemaker_session=sagemaker_session
  )

   # Deploy the model as a serverless endpoint
  endpoint_name = os.environ.get('SAGEMAKER_ENDPOINT_NAME', 'controlnet-qr-endpoint')
  print(f"Deploying model to endpoint: {endpoint_name}")

  output_path = f"s3://{bucket}/controlnet-qr-output/"
  print(f"Async inference output will be stored at: {output_path}")

  predictor = model.deploy(
    endpoint_name=endpoint_name,
    serverless_inference_config=serverless_config,
    async_inference_config={
        "output_path": output_path
    }
  )

  print(f"Deployed SageMaker serverless endpoint: controlnet-qr-endpoint")
  print(f"Async inference output will be stored in {output_path}")
  
  return predictor

if __name__ == "__main__":
  deploy_sagemaker_endpoint()