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
import time
import sagemaker
from sagemaker.model import Model
from sagemaker.async_inference.async_inference_config import AsyncInferenceConfig


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
  # First check if the repository exists
  try:
    ecr_client = boto3.client('ecr')
    ecr_client.describe_repositories(repositoryNames=[repository_name])
    print(f"Repository {repository_name} already exists")
  except ecr_client.exceptions.RepositoryNotFoundException:
    print(f"Repository {repository_name} not found, creating it...")
    ecr_client.create_repository(repositoryName=repository_name)

  # Get the repository URI
  response = ecr_client.describe_repositories(repositoryNames=[repository_name])
  repository_uri = response['repositories'][0]['repositoryUri']
  print(f"Repository URI: {repository_uri}")

  # Build the Docker image
  print("Building Docker image...")
  import subprocess
  build_cmd = [
    "docker", "build", 
    "--platform", "linux/amd64",
    "-t", repository_name,
    "-f", "./apps/controlnet/Dockerfile",
    "./apps/controlnet"
  ]
  subprocess.run(build_cmd, check=True)

  # Tag the image
  tag_cmd = ["docker", "tag", f"{repository_name}:latest", f"{repository_uri}:latest"]
  subprocess.run(tag_cmd, check=True)

  # Push the image to ECR
  print("Pushing Docker image to ECR...")
  push_cmd = ["docker", "push", f"{repository_uri}:latest"]
  subprocess.run(push_cmd, check=True)

  # Set the image URI for SageMaker
  image_uri = f"{repository_uri}:latest"
  print(f"Image URI: {image_uri}")

   # Create the model
  model_name = os.environ.get("SAGEMAKER_MODEL_NAME", "controlnet-qr-model")
  print(f"Creating SageMaker model: {model_name}")

  model = Model(
    image_uri=image_uri,
    role=role,
    name=model_name,
    sagemaker_session=sagemaker_session
  )

   # Deploy the model
  endpoint_name = os.environ.get('SAGEMAKER_ENDPOINT_NAME', 'controlnet-qr-endpoint')
  timestamp = int(time.time())
  endpoint_config_name = f"{endpoint_name}-config-{timestamp}"
  print(f"Creating endpoint config: {endpoint_config_name}")
  print(f"Deploying model to endpoint: {endpoint_name}")

  instance_type = os.environ.get("SAGEMAKER_INSTANCE_TYPE", "ml.g5.2xlarge")

  output_path = f"s3://{bucket}/controlnet-qr-output/"
  print(f"Async inference output will be stored at: {output_path}")

  # Create an AsyncInferenceConfig object
  async_config = AsyncInferenceConfig(
      output_path=output_path,
      max_concurrent_invocations_per_instance=1,
  )

  sm_client = boto_session.client('sagemaker')

  # Check if endpoint exists and delete it
  try:
    print(f"Checking if endpoint {endpoint_name} already exists...")
    sm_client.describe_endpoint(EndpointName=endpoint_name)
    print(f"Deleting existing endpoint: {endpoint_name}")
    sm_client.delete_endpoint(EndpointName=endpoint_name)
    
    # Wait for endpoint deletion to complete
    print("Waiting for endpoint deletion... (this may take a few minutes)")
    waiter = sm_client.get_waiter('endpoint_deleted')
    waiter.wait(EndpointName=endpoint_name)
    print(f"Endpoint {endpoint_name} successfully deleted.")
  except sm_client.exceptions.ClientError:
    print(f"Endpoint {endpoint_name} doesn't exist, nothing to delete.")

  # Check if endpoint config exists and delete it
  try:
    print(f"Checking if endpoint config {endpoint_config_name} already exists...")
    sm_client.describe_endpoint_config(EndpointConfigName=endpoint_name)
    print(f"Deleting existing endpoint config: {endpoint_name}")
    sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
  except sm_client.exceptions.ClientError:
    print(f"Endpoint config {endpoint_name} doesn't exist, nothing to delete.")

  predictor = model.deploy(
    endpoint_name=endpoint_name,
    instance_type=instance_type,
    initial_instance_count=1,
    async_inference_config=async_config,
    endpoint_config_name=endpoint_config_name
  )

  setup_autoscaling(endpoint_name)

  print(f"Deployed SageMaker serverless endpoint: controlnet-qr-endpoint")
  print(f"Async inference output will be stored in {output_path}")
  
  return predictor

def setup_autoscaling(endpoint_name):
  """Configure auto-scaling for the endpoint to scale down to zero when idle"""
  import boto3
  
  app_autoscaling = boto3.client('application-autoscaling')
  cloudwatch = boto3.client('cloudwatch')
  
  # Define the resource ID for the endpoint
  resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"
  
  # Register the endpoint as a scalable target
  app_autoscaling.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId=resource_id,
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=0,
    MaxCapacity=2
  )
  
  # Scaling policy based on queue backlog
  response = app_autoscaling.put_scaling_policy(
    PolicyName="Invocations-ScalingPolicy",
    ServiceNamespace="sagemaker",
    ResourceId=resource_id,
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
      "TargetValue": 1.0,
      "CustomizedMetricSpecification": {
        "MetricName": "ApproximateBacklogSizePerInstance",
        "Namespace": "AWS/SageMaker",
        "Dimensions": [{"Name": "EndpointName", "Value": endpoint_name}],
        "Statistic": "Average",
      },
      "ScaleInCooldown": 600,
      "ScaleOutCooldown": 300
    }
  )
  
  # Policy to wake up the endpoint from zero
  response = app_autoscaling.put_scaling_policy(
    PolicyName="HasBacklogWithoutCapacity-ScalingPolicy",
    ServiceNamespace="sagemaker",
    ResourceId=resource_id,
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="StepScaling",
    StepScalingPolicyConfiguration={
      "AdjustmentType": "ChangeInCapacity",
      "MetricAggregationType": "Average",
      "Cooldown": 300,
      "StepAdjustments": [
        {
          "MetricIntervalLowerBound": 0,
          "ScalingAdjustment": 1
        }
      ]
    }
  )
  
  # CloudWatch alarm to monitor for requests when at zero capacity
  cloudwatch.put_metric_alarm(
    AlarmName="HasBacklogWithoutCapacity-ScalingPolicy",
    MetricName='HasBacklogWithoutCapacity',
    Namespace='AWS/SageMaker',
    Statistic='Average',
    EvaluationPeriods=2,
    DatapointsToAlarm=2,
    Threshold=1,
    ComparisonOperator='GreaterThanOrEqualToThreshold',
    TreatMissingData='missing',
    Dimensions=[
      {'Name':'EndpointName', 'Value': endpoint_name},
    ],
    Period=60,
    AlarmActions=[response['PolicyARN']],
  )

if __name__ == "__main__":
  deploy_sagemaker_endpoint()