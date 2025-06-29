# HADM Docker Development Environment

This document describes how to use the Docker development environment for the Human Artifact Detection Model (HADM) project.

## Prerequisites

- Docker with GPU support (NVIDIA Docker runtime)
- NVIDIA drivers compatible with CUDA 11.6
- VS Code with Dev Containers extension (for dev container usage)
- At least 16GB of available disk space

## Quick Start with VS Code Dev Containers

1. **Open in VS Code**: Open this project folder in VS Code
2. **Reopen in Container**: When prompted, click "Reopen in Container" or use Command Palette → "Dev Containers: Reopen in Container"
3. **Wait for Build**: The first build will take 15-30 minutes to install all dependencies
4. **Start Developing**: Once built, you'll have a full development environment with GPU support

## Manual Docker Usage

### Build the Image
```bash
docker build -t hadm:latest .
```

### Run with Docker Compose
```bash
# Start development environment
docker-compose up -d hadm-dev

# Start Jupyter Lab
docker-compose up -d hadm-jupyter

# Start TensorBoard
docker-compose up -d hadm-tensorboard

# Access the development container
docker-compose exec hadm-dev bash
```

### Run with Docker directly
```bash
docker run -it --gpus all \
  --shm-size=8g \
  -v $(pwd):/workspace/code \
  -v hadm-datasets:/workspace/datasets \
  -v hadm-models:/workspace/pretrained_models \
  -v hadm-outputs:/workspace/outputs \
  -v hadm-cache:/workspace/cache \
  -p 8888:8888 -p 6006:6006 \
  hadm:latest
```

## Environment Structure

The container provides the following workspace structure:

```
/workspace/
├── code/                    # Project source code (bind mounted)
├── datasets/               # Dataset storage (persistent volume)
├── pretrained_models/      # Model weights (persistent volume)
├── outputs/               # Training outputs (persistent volume)
└── cache/                 # Evaluation cache (persistent volume)
```

## Available Services

### Development Environment
- **Container**: `hadm-dev`
- **Purpose**: Interactive development with full GPU access
- **Access**: `docker-compose exec hadm-dev bash`

### Jupyter Lab
- **Container**: `hadm-jupyter`
- **URL**: http://localhost:8888
- **Purpose**: Interactive notebook development
- **Start**: `docker-compose up -d hadm-jupyter`

### TensorBoard
- **Container**: `hadm-tensorboard`
- **URL**: http://localhost:6006
- **Purpose**: Training visualization
- **Start**: `docker-compose up -d hadm-tensorboard`

## Setting Up Datasets and Models

After starting the container, you'll need to download the required datasets and models:

### 1. Download Human Artifact Dataset
```bash
# Inside the container
cd /workspace/datasets
wget "https://www.dropbox.com/scl/fi/823u0q183t0oamaxcg4qv/human_artifact_dataset.zip?rlkey=nbz2vvg14av43h66ac1p7kbvb&st=9e8q0gcf&dl=0" -O human_artifact_dataset.zip
unzip human_artifact_dataset.zip
```

### 2. Download Pretrained Models
```bash
# Inside the container
cd /workspace/pretrained_models

# EVA-02-L weights
wget "https://huggingface.co/Yuxin-CV/EVA-02/resolve/main/eva02/det/eva02_L_coco_det_sys_o365.pth"

# HADM-L weights
wget "https://www.dropbox.com/scl/fi/zwasvod906x1akzinnj3i/HADM-L_0249999.pth?rlkey=bqz5517tm8yt8l6ngzne4xejx&st=k1a1gzph&dl=0" -O HADM-L_0249999.pth

# HADM-G weights
wget "https://www.dropbox.com/scl/fi/bzj1m8p4cvm2vg4mai6uj/HADM-G_0249999.pth?rlkey=813x6wraigivc6qx02aut9p2r&st=n8rnb47r&dl=0" -O HADM-G_0249999.pth
```

### 3. Set Up Additional Real Human Images (Optional)
Follow the instructions in the main README to download and set up additional training datasets.

## Running Examples

### Demo Inference
```bash
# Local Human Artifact Detection
python tools/lazyconfig_train_net.py --num-gpus 1 --inference \
    --config-file projects/ViTDet/configs/eva2_o365_to_coco/demo_local.py \
    train.output_dir=./outputs/demo_local \
    train.init_checkpoint=pretrained_models/HADM-L_0249999.pth \
    dataloader.train.total_batch_size=1 \
    train.model_ema.enabled=True \
    train.model_ema.use_ema_weights_for_eval_only=True \
    inference.input_dir=demo/images \
    inference.output_dir=demo/outputs/result_local
```

### Model Evaluation
```bash
# Evaluate HADM-L on all domains
python tools/lazyconfig_train_net.py --num-gpus 1 --eval-only \
    --config-file projects/ViTDet/configs/eva2_o365_to_coco/eva02_large_local.py \
    train.output_dir=./outputs/eva02_large_local/250k_on_all_val \
    train.init_checkpoint=pretrained_models/HADM-L_0249999.pth \
    dataloader.evaluator.output_dir=cache/large_local_human_artifact_ALL_val/250k_on_all_val \
    dataloader.evaluator.dataset_name=local_human_artifact_val_ALL \
    dataloader.test.dataset.names=local_human_artifact_val_ALL \
    dataloader.train.total_batch_size=1 \
    train.model_ema.enabled=True \
    train.model_ema.use_ema_weights_for_eval_only=True
```

## Troubleshooting

### GPU Not Available
- Ensure NVIDIA Docker runtime is installed
- Check GPU access: `nvidia-smi` inside container
- Verify Docker GPU support: `docker run --rm --gpus all nvidia/cuda:11.6-base nvidia-smi`

### Out of Memory
- Reduce batch size in training commands
- Increase shared memory: `--shm-size=16g`
- Monitor GPU memory: `watch -n 1 nvidia-smi`

### Permission Issues
- Ensure USER_ID and GROUP_ID match your host user
- Set environment variables: `export USER_ID=$(id -u) GROUP_ID=$(id -g)`

### Build Issues
- Clear Docker cache: `docker system prune -a`
- Rebuild without cache: `docker-compose build --no-cache`

## Development Tips

1. **Live Code Editing**: Changes to source code are immediately available in the container
2. **Persistent Data**: Datasets, models, and outputs persist across container restarts
3. **Multiple Terminals**: Open multiple VS Code terminals for parallel tasks
4. **Jupyter Integration**: Use Jupyter notebooks for interactive development
5. **TensorBoard Monitoring**: Monitor training progress in real-time

## Environment Variables

- `DETECTRON2_DATASETS`: Points to `/workspace/datasets`
- `PYTHONPATH`: Includes `/workspace/code`
- `CUDA_VISIBLE_DEVICES`: Set to `all` for GPU access
- `NVIDIA_VISIBLE_DEVICES`: Set to `all` for GPU access

## Volume Management

Persistent volumes are used for:
- `hadm-datasets`: Dataset storage
- `hadm-models`: Pretrained model weights
- `hadm-outputs`: Training outputs and results
- `hadm-cache`: Evaluation cache files

To clean up volumes:
```bash
docker-compose down -v  # Remove all volumes
docker volume prune     # Remove unused volumes
