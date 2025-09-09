#!/bin/bash

# print the commands
set -o xtrace

# Run docker image as an argument but use triplane-cfd by default
DATASET_PATH=${1:-"datasets"}
DOCKER_IMAGE=${2:-"mosaic3d"}

USER=$(whoami)

# Mount the current path to /workspace
docker run \
    --gpus all \
    --shm-size=32g \
    -it \
    --ipc host \
    --net host \
    --name mosaic3d \
    -v "/home/${USER}:/root" \
    -v "$(pwd):/workspace" \
    -v "${DATASET_PATH}:/datasets" \
    --workdir /workspace \
    --device=/dev/nvidiactl \
    --device=/dev/nvidia0 \
    --device=/dev/nvidia1 \
    --device=/dev/nvidia2 \
    --device=/dev/nvidia3 \
    --device=/dev/nvidia4 \
    --device=/dev/nvidia5 \
    --device=/dev/nvidia6 \
    --device=/dev/nvidia7 \
    --device=/dev/nvidia-modeset \
    --device=/dev/nvidia-uvm \
    --device=/dev/nvidia-uvm-tools \
    "$DOCKER_IMAGE" \
    /bin/zsh
