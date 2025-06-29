# Use CUDA 11.6 with cuDNN 8 to match PyTorch 1.12.1+cu116 requirement
FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies with tmpfs for faster package operations
RUN --mount=type=tmpfs,target=/var/lib/apt/lists \
    --mount=type=tmpfs,target=/var/cache/apt \
    apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    ca-certificates

# Create a non-root user first
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} developer && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash developer


# Switch to non-root user
USER developer
WORKDIR /workspace

# Install Miniconda to user home directory
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /home/developer/miniconda && \
    rm /tmp/miniconda.sh

# Update PATH to include conda binaries
ENV PATH="/home/developer/miniconda/bin:${PATH}"

# Create conda environment following README exactly with tmpfs for faster package operations
RUN --mount=type=tmpfs,target=/tmp \
    conda create --name hadm python=3.8 -y

# Make RUN commands use the conda environment
SHELL ["conda", "run", "-n", "hadm", "/bin/bash", "-c"]

# Install PyTorch with CUDA 11.6 support (following README exactly) using tmpfs for pip cache
RUN --mount=type=tmpfs,target=/tmp \
    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# Install cryptography (as specified in README) using tmpfs
RUN --mount=type=tmpfs,target=/tmp \
    pip install cryptography

# Copy requirements and install dependencies using tmpfs for pip cache
COPY --chown=developer:developer requirements.txt /workspace/
RUN --mount=type=tmpfs,target=/tmp \
    pip install -r requirements.txt

# Install xformers from git (following README exactly) using tmpfs for compilation
# This is the most compilation-heavy step, so tmpfs will provide significant speedup
RUN --mount=type=tmpfs,target=/tmp \
    rm -rf /home/developer/.cache && \
    mkdir /tmp/developer.cache && ln -sf /tmp/developer.cache /home/developer/.cache && \
    pip install -v -U git+https://github.com/facebookresearch/xformers.git@v0.0.18#egg=xformers && \
    pip install mmcv==1.7.1 openmim && \
    mim install mmcv-full

# Copy the project source code
COPY --chown=developer:developer . /workspace/

RUN python -m pip install -e .

USER root

RUN apt-get update && apt-get install -y libgl1 vim libglib2.0-0

USER developer

# Ensure conda environment is activated by default
RUN conda init bash
