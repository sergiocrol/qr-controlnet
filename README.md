# QR-ControlNet Project - Local Testing with Cloud-Ready Code

This document explains how to test your SageMaker-ready code locally without adding local-only code paths, ensuring your code behaves the same way in both environments.

## Project Structure Improvements

The code has been reorganized to:

1. **Keep SageMaker compatibility while enabling local testing**

   - Uses the same entry points and API endpoints
   - Maintains the same directory structure expected by SageMaker

2. **Clearer separation of concerns**

   - Flask app initialization and routes better organized
   - SageMaker routes separated from regular API endpoints
   - Common inference code shared between both paths

3. **Consistent environment variables**
   - Same configuration works locally and in the cloud

## Key Files Added/Modified

- **Dockerfile**: Updated to be cloud-ready but locally testable
- **docker-compose.sagemaker-local.yml**: New file for local SageMaker testing
- **Makefile**: Enhanced with SageMaker-specific commands
- **serve**: SageMaker entrypoint script
- **app/**init**.py**: Refactored to handle both regular API and SageMaker endpoints
- **test_client**: New directory with scripts to test the SageMaker container

## How to Test Locally

### 1. Build the SageMaker-compatible container

```bash
make build-sagemaker-local
```

This builds a container that's identical to what would run in SageMaker.

### 2. Run the container locally

```bash
make run-sagemaker-local
```

This starts the container with the same entrypoint and environment variables as SageMaker.

### 3. Test the endpoints

```bash
# Test the health endpoint
make ping-sagemaker-local

# Test image generation
make test-sagemaker-local
```

### 4. Using the test client

```bash
# Run the test client
python apps/controlnet/test_client/test_sagemaker_local.py \
  --prompt "A QR code in the style of Picasso" \
  --steps 10
```

## Deployment to SageMaker

When you're ready to deploy to AWS SageMaker:

```bash
make deploy-sagemaker
```

This will build and push your container to ECR and create a SageMaker endpoint.

## Understanding the SageMaker Integration

SageMaker expects specific endpoints and directory structures:

1. **/ping**: Health check endpoint (must return 200 OK when ready)
2. **/invocations**: Inference endpoint (accepts POST requests with JSON payload)
3. **Directories**: SageMaker uses specific directories like /opt/ml/model, /opt/ml/input, etc.

Our setup ensures these requirements are met while keeping the code clean and maintainable.

## Benefits of This Approach

- **Consistency**: Same code runs locally and in the cloud
- **Confidence**: What works locally will work in SageMaker
- **Efficiency**: Faster development cycle with local testing
- **Cost savings**: Debug locally before deploying to AWS

## Next Steps

1. Complete testing locally with your QR code generation
2. Deploy to SageMaker when ready
3. Set up CI/CD for automated testing and deployment
