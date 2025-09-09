# Define base image.  # Use 12.1.0 for NVIDIA warp
FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# metainformation
LABEL org.opencontainers.image.version = "0.0.1"

# CUDA architectures.
## A5500 is 86, A100 is 80, V100 is 70, T4 is 75
ARG CUDA_ARCHITECTURES=86;80;
## TORCH_CUDA_ARCH_LIST for torch_scatter
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6"

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
    git-lfs \
    tig \
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

# Git lfs
RUN git lfs install

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
RUN pip3 install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Install MinkowskiEngine manually for Segment3D
# this is a patch for MinkowskiEngine to work with CUDA 12.1 (https://github.com/NVIDIA/MinkowskiEngine/issues/543)
RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git /tmp/MinkowskiEngine \
    && cd /tmp/MinkowskiEngine \
    && sed -i '31i #include <thrust/execution_policy.h>' ./src/convolution_kernel.cuh \
    && sed -i '39i #include <thrust/unique.h>\n#include <thrust/remove.h>' ./src/coordinate_map_gpu.cu \
    && sed -i '38i #include <thrust/execution_policy.h>\n#include <thrust/reduce.h>\n#include <thrust/sort.h>' ./src/spmm.cu \
    && sed -i '38i #include <thrust/execution_policy.h>' ./src/3rdparty/concurrent_unordered_map.cuh \
    && python3 setup.py install --force_cuda --blas=openblas \
    && cd - \
    && rm -rf /tmp/MinkowskiEngine

# Install Segmentator and pointnet2 manually for Segment3D
WORKDIR /tmp
RUN git clone https://github.com/LeapLabTHU/Segment3D.git \
    && cd Segment3D/third_party/Segmentator \
    && make \
    && cp segmentator /usr/local/bin/segmentator \
    && cd ../pointnet2 \
    && python3 setup.py install \
    && cd /tmp \
    && rm -rf /tmp/Segment3D

# Install pointops manually for faster FPS sampling
WORKDIR /tmp
RUN git clone https://github.com/Pointcept/Pointcept.git \
    && cd Pointcept/libs/pointops \
    && python3 setup.py install \
    && cd /tmp \
    && rm -rf /tmp/Pointcept

# Debugging tools
RUN pip3 install ipdb

RUN chsh -s /usr/bin/bash user
