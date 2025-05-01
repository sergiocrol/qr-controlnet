# Makefile for QR-AI project with SageMaker compatibility

# Default values
PLATFORM ?= auto
SERVICE ?= all
ENV ?= prod

# Create necessary directories
init:
	mkdir -p shared_results

# Build all images for production
build: init
	docker-compose build

# Build all images for development
build-dev: init
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml build

# Build SageMaker-compatible container for local testing
build-sagemaker-local: init
	docker-compose -f docker-compose.sagemaker-local.yml build

# Run the SageMaker-compatible container locally
run-sagemaker-local: init
	docker-compose -f docker-compose.sagemaker-local.yml up

# Run the SageMaker-compatible container locally in detached mode
run-sagemaker-local-detached: init
	docker-compose -f docker-compose.sagemaker-local.yml up -d

# Run all services in production mode
up: init
	DEVICE=$(PLATFORM) docker-compose up

# Run all services in detached mode
up-detached: init
	DEVICE=$(PLATFORM) docker-compose up -d

# Run all services in development mode
dev: init
	DEVICE=$(PLATFORM) docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Run all services in development mode (detached)
dev-detached: init
	DEVICE=$(PLATFORM) docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Test SageMaker inference locally
test-sagemaker-local:
	curl -X POST http://localhost:8080/invocations \
		-H "Content-Type: application/json" \
		-d '{"prompt": "A futuristic cityscape with neon lights", "num_inference_steps": 10}'

# Test SageMaker health check locally
ping-sagemaker-local:
	curl http://localhost:8080/ping

# Generate a QR code using the SageMaker container
test-qr-sagemaker:
	curl -X POST http://localhost:8080/invocations \
		-H "Content-Type: application/json" \
		-d '{"prompt": "A giant whale flying in the sky", "num_inference_steps": 10}'

# Run all services in the project
run-all: init
	DEVICE=$(PLATFORM) docker-compose up

# Stop all services
down:
	docker-compose down

# Stop SageMaker local container
down-sagemaker-local:
	docker-compose -f docker-compose.sagemaker-local.yml down

# Stop and remove volumes (clean everything)
clean:
	docker-compose down -v
	docker-compose -f docker-compose.sagemaker-local.yml down -v
	rm -rf shared_results/*

# Rebuild and restart SageMaker local container
restart-sagemaker-local:
	docker-compose -f docker-compose.sagemaker-local.yml up -d --build

# Show container logs for SageMaker local
logs-sagemaker-local:
	docker-compose -f docker-compose.sagemaker-local.yml logs -f

# Enter shell in the SageMaker local container
shell-sagemaker-local:
	docker-compose -f docker-compose.sagemaker-local.yml exec sagemaker-local bash

# Deploy to AWS SageMaker using the deployment script
deploy-sagemaker:
	cd ./apps/controlnet && python deploy_sagemaker.py

# Build for SageMaker deployment
sagemaker-build-deploy:
	cd ./apps/controlnet && \
	docker build -t controlnet-qr-sagemaker:latest \
		--platform=linux/amd64 \
		-f Dockerfile .

# Test AWS SageMaker async endpoint (requires AWS CLI setup)
test-sagemaker-async:
	aws sagemaker-runtime invoke-endpoint \
		--endpoint-name controlnet-qr-endpoint \
		--content-type application/json \
		--body '{"prompt": "A futuristic cityscape with neon lights", "num_inference_steps": 10}' \
		output.json

# Force clean and rebuild the SageMaker container
sagemaker-rebuild:
	docker-compose -f docker-compose.sagemaker-local.yml rm -f sagemaker-local
	docker-compose -f docker-compose.sagemaker-local.yml build --no-cache sagemaker-local
	
# Show help
help:
	@echo "QR-AI Makefile Commands:"
	@echo ""
	@echo "== Local SageMaker Testing =="
	@echo "make build-sagemaker-local    - Build SageMaker-compatible container for local testing"
	@echo "make run-sagemaker-local      - Run the SageMaker container locally"
	@echo "make test-sagemaker-local     - Test SageMaker inference locally"
	@echo "make ping-sagemaker-local     - Test SageMaker health check locally"
	@echo "make test-qr-sagemaker        - Generate a QR code using the SageMaker container"
	@echo ""
	@echo "== Docker Compose Commands =="
	@echo "make build                    - Build all services for production"
	@echo "make build-dev                - Build all services for development"
	@echo "make up                       - Run all services in production mode"
	@echo "make up-detached              - Run all services in production mode (background)"
	@echo "make dev                      - Run all services in development mode"
	@echo "make dev-detached             - Run all services in development mode (background)"
	@echo "make down                     - Stop all services"
	@echo "make clean                    - Clean everything (volumes, etc.)"
	@echo ""
	@echo "== AWS SageMaker Deployment =="
	@echo "make deploy-sagemaker         - Deploy to AWS SageMaker"
	@echo "make sagemaker-build-deploy   - Build for SageMaker deployment"
	@echo "make test-sagemaker-async     - Test AWS SageMaker async endpoint"
	@echo ""
	@echo "== Debugging =="
	@echo "make logs-sagemaker-local     - Show logs for SageMaker local container"
	@echo "make shell-sagemaker-local    - Shell into SageMaker local container"
	@echo "make sagemaker-rebuild        - Force rebuild SageMaker container"
	@echo ""
	@echo "Options (set as environment variables):"
	@echo "PLATFORM=cpu|mps|cuda|auto    - Set the computation platform"
	@echo ""
	@echo "Examples:"
	@echo "make PLATFORM=cpu run-sagemaker-local  - Run with CPU acceleration"
	@echo "make build-sagemaker-local test-sagemaker-local - Build and test locally"

# Default target
.DEFAULT_GOAL := help