# Define base image.  # Use 11.7.1 for open3d==0.18 which is compiled with torch 2 and cuda 11.7.1
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# metainformation
LABEL org.opencontainers.image.version = "0.0.1"
LABEL org.opencontainers.image.source = "https://gitlab.nvidia.com:3dmmllm/openvocab-3d"

# CUDA architectures.
## A100 is 80, V100 is 70, T4 is 75
ARG CUDA_ARCHITECTURES=86;80;75;70;
## TORCH_CUDA_ARCH_LIST for torch_scatter
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6"

# Set environment variables.
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive
## Set timezone as it is required by some packages.
ENV TZ=America/Los_Angeles
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"

# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    openssh-server \
    cmake \
    curl \
    git \
    zsh \
    tmux \
    vim \
    libatlas-base-dev \
    libhdf5-dev \
    libcgal-dev \
    libeigen3-dev \
    libfreeimage-dev \
    libgflags-dev \
    libegl1 \
    libgl1 \
    libgomp1 \
    libopenblas-dev \
    protobuf-compiler \
    python-is-python3 \
    python3-dev \
    python3-pip \
    sudo \
    vim-tiny \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Create non root user and setup environment.
RUN useradd -m -d /home/user -g root -G sudo -u 1000 user
RUN usermod -aG sudo user
# Set user password
RUN echo "user:user" | chpasswd
# Ensure sudo group users are not asked for a password when using sudo command by amending sudoers file
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Install packages.
RUN pip3 install -U pip

# Install torch first for torch scatter installation
RUN pip3 install torch==2.0.1 torchvision --extra-index-url https://download.pytorch.org/whl/cu117
COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Install torch_scatter manually
RUN pip3 install --no-index torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html

RUN chsh -s /usr/bin/bash user
