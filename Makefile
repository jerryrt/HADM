.PHONY: help build up down dev jupyter tensorboard clean logs shell test

# Default target
help:
	@echo "HADM Docker Development Environment"
	@echo "=================================="
	@echo ""
	@echo "Available commands:"
	@echo "  build        - Build the Docker image"
	@echo "  up           - Start all services"
	@echo "  down         - Stop all services"
	@echo "  dev          - Start development container"
	@echo "  jupyter      - Start Jupyter Lab service"
	@echo "  tensorboard  - Start TensorBoard service"
	@echo "  shell        - Open shell in development container"
	@echo "  logs         - Show logs from all services"
	@echo "  clean        - Clean up containers and images"
	@echo "  clean-all    - Clean up everything including volumes"
	@echo "  test         - Test GPU access in container"
	@echo ""
	@echo "VS Code Dev Container:"
	@echo "  Open this folder in VS Code and select 'Reopen in Container'"

# Build the Docker image
build:
	@echo "Building HADM Docker image..."
	docker-compose build

# Start all services
up:
	@echo "Starting all HADM services..."
	docker-compose up -d

# Stop all services
down:
	@echo "Stopping all HADM services..."
	docker-compose down

# Start development container
dev:
	@echo "Starting HADM development container..."
	docker-compose up -d hadm-dev

# Start Jupyter Lab
jupyter:
	@echo "Starting Jupyter Lab..."
	@echo "Access at: http://localhost:8888"
	docker-compose up -d hadm-jupyter

# Start TensorBoard
tensorboard:
	@echo "Starting TensorBoard..."
	@echo "Access at: http://localhost:6006"
	docker-compose up -d hadm-tensorboard

# Open shell in development container
shell:
	@echo "Opening shell in HADM development container..."
	docker-compose exec hadm-dev bash

# Show logs
logs:
	docker-compose logs -f

# Clean up containers and images
clean:
	@echo "Cleaning up HADM containers and images..."
	docker-compose down --rmi all
	docker system prune -f

# Clean up everything including volumes
clean-all:
	@echo "Cleaning up everything including volumes..."
	docker-compose down -v --rmi all
	docker system prune -a -f
	docker volume prune -f

# Test GPU access
test:
	@echo "Testing GPU access in container..."
	docker-compose run --rm hadm-dev conda run -n hadm python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Quick demo run
demo:
	@echo "Running demo inference (requires pretrained models)..."
	docker-compose exec hadm-dev conda run -n hadm python tools/lazyconfig_train_net.py --num-gpus 1 --inference \
		--config-file projects/ViTDet/configs/eva2_o365_to_coco/demo_local.py \
		train.output_dir=./outputs/demo_local \
		train.init_checkpoint=pretrained_models/HADM-L_0249999.pth \
		dataloader.train.total_batch_size=1 \
		train.model_ema.enabled=True \
		train.model_ema.use_ema_weights_for_eval_only=True \
		inference.input_dir=demo/images \
		inference.output_dir=demo/outputs/result_local

# Setup datasets and models
setup:
	@echo "Setting up datasets and models..."
	@echo "This will download large files. Make sure you have sufficient disk space and bandwidth."
	@echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
	@sleep 5
	docker-compose exec hadm-dev bash -c "cd /workspace/datasets && wget 'https://www.dropbox.com/scl/fi/823u0q183t0oamaxcg4qv/human_artifact_dataset.zip?rlkey=nbz2vvg14av43h66ac1p7kbvb&st=9e8q0gcf&dl=0' -O human_artifact_dataset.zip && unzip human_artifact_dataset.zip"
	docker-compose exec hadm-dev bash -c "cd /workspace/pretrained_models && wget 'https://huggingface.co/Yuxin-CV/EVA-02/resolve/main/eva02/det/eva02_L_coco_det_sys_o365.pth' && wget 'https://www.dropbox.com/scl/fi/zwasvod906x1akzinnj3i/HADM-L_0249999.pth?rlkey=bqz5517tm8yt8l6ngzne4xejx&st=k1a1gzph&dl=0' -O HADM-L_0249999.pth && wget 'https://www.dropbox.com/scl/fi/bzj1m8p4cvm2vg4mai6uj/HADM-G_0249999.pth?rlkey=813x6wraigivc6qx02aut9p2r&st=n8rnb47r&dl=0' -O HADM-G_0249999.pth"
	@echo "Setup complete!"

# Show status
status:
	@echo "HADM Docker Services Status:"
	@echo "============================"
	docker-compose ps
	@echo ""
	@echo "Docker Volumes:"
	@echo "==============="
	docker volume ls | grep hadm
