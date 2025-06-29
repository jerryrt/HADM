# Use CUDA 11.6 with cuDNN 8 to match PyTorch 1.12.1+cu116 requirement
FROM nvidia/cuda:11.6-cudnn8-devel-ubuntu20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Add conda to PATH
ENV PATH="/opt/conda/bin:${PATH}"

# Create a non-root user
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} developer && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash developer && \
    chown -R developer:developer /opt/conda

# Switch to non-root user
USER developer
WORKDIR /workspace

# Create conda environment following README exactly
RUN conda create --name hadm python=3.8 -y

# Make RUN commands use the conda environment
SHELL ["conda", "run", "-n", "hadm", "/bin/bash", "-c"]

# Install PyTorch with CUDA 11.6 support (following README exactly)
RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# Install cryptography (as specified in README)
RUN pip install cryptography

# Copy requirements and install dependencies
COPY --chown=developer:developer requirements.txt /workspace/
RUN pip install -r requirements.txt

# Install xformers from git (following README exactly)
RUN pip install -v -U git+https://github.com/facebookresearch/xformers.git@v0.0.18#egg=xformers

# Install mmcv and openmim (following README exactly)
RUN pip install mmcv==1.7.1 openmim && \
    mim install mmcv-full

# Copy the project source code
COPY --chown=developer:developer . /workspace/code/

# Install the project in editable mode (following README exactly)
WORKDIR /workspace/code
RUN python -m pip install -e .

# Set environment variables
ENV DETECTRON2_DATASETS=/workspace/datasets
ENV PYTHONPATH=/workspace/code:$PYTHONPATH

# Create necessary directories
RUN mkdir -p /workspace/datasets \
             /workspace/pretrained_models \
             /workspace/outputs \
             /workspace/cache

# Set working directory back to workspace root
WORKDIR /workspace

# Ensure conda environment is activated by default
RUN echo "conda activate hadm" >> ~/.bashrc

# Set default command
CMD ["conda", "run", "-n", "hadm", "/bin/bash"]
