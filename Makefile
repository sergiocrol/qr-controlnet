# Makefile for QR-AI project with multi-stage Docker support

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

# Build a specific service for production
build-%: init
	docker-compose build $*

# Build a specific service for development
build-dev-%: init
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml build $*

# Run all services in production mode
up: init
	PREFERRED_DEVICE=$(PLATFORM) docker-compose up

# Run all services in detached mode
up-detached: init
	PREFERRED_DEVICE=$(PLATFORM) docker-compose up -d

# Run all services in development mode
dev: init
	PREFERRED_DEVICE=$(PLATFORM) docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Run all services in development mode (detached)
dev-detached: init
	PREFERRED_DEVICE=$(PLATFORM) docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Run a specific service in production mode
run-%: init
	PREFERRED_DEVICE=$(PLATFORM) docker-compose up $*

# Run a specific service in development mode
dev-%: init
	PREFERRED_DEVICE=$(PLATFORM) docker-compose -f docker-compose.yml -f docker-compose.dev.yml up $*

# Stop all services
down:
	docker-compose down

# Stop and remove volumes (clean everything)
clean:
	docker-compose down -v
	rm -rf shared_results/*

# Rebuild and restart a specific service (useful during development)
restart-%:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build $*

# Show container logs
logs:
	docker-compose logs -f

# Show logs for a specific service
logs-%:
	docker-compose logs -f $*

# Enter shell in a running container
shell-%:
	docker-compose exec $* sh

# Run tests
test:
	@echo "Running tests..."
	# Add your test commands here

# Generate a QR code with the test script (once containers are running)
test-qr:
	docker-compose exec controlnet python test_api.py --image qrs/qr.png --prompt "A giant whale flying in the sky" --size 512 --steps 10 --device $(PLATFORM)

# Docker containers on macOS cannot access the Mac's MPS hardware acceleration. For development in Mac better test without DOCKER
test-qr-native:
	cd ./apps/controlnet && PREFERRED_DEVICE=$(PLATFORM) python test_api.py --image qrs/qr.png --prompt "A giant whale flying in the sky" --size 512 --steps 10 --device $(PLATFORM)

# Show help
help:
	@echo "QR-AI Makefile Commands:"
	@echo "make build                 - Build all services for production"
	@echo "make build-dev             - Build all services for development"
	@echo "make build-controlnet      - Build only the controlnet service for production"
	@echo "make build-dev-controlnet  - Build only the controlnet service for development"
	@echo "make up                    - Run all services in production mode"
	@echo "make up-detached           - Run all services in production mode (background)"
	@echo "make dev                   - Run all services in development mode"
	@echo "make dev-detached          - Run all services in development mode (background)"
	@echo "make run-controlnet        - Run only the controlnet service in production mode"
	@echo "make dev-controlnet        - Run only the controlnet service in development mode"
	@echo "make restart-controlnet    - Rebuild and restart the controlnet service"
	@echo "make down                  - Stop all services"
	@echo "make clean                 - Clean everything (volumes, etc.)"
	@echo "make logs                  - Show logs for all services"
	@echo "make logs-controlnet       - Show logs for the controlnet service"
	@echo "make shell-controlnet      - Open a shell in the controlnet container"
	@echo "make test-qr               - Run a test QR code generation"
	@echo ""
	@echo "Options (set as environment variables):"
	@echo "PLATFORM=cpu|mps|cuda|auto - Set the computation platform"
	@echo ""
	@echo "Examples:"
	@echo "make PLATFORM=cpu up       - Run with CPU acceleration"
	@echo "make build-dev dev-controlnet - Build all for dev and run only controlnet"

# Default target
.DEFAULT_GOAL := help