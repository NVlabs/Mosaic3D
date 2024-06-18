#!/bin/bash

# print the commands
set -o xtrace

# Run docker image as an argument but use triplane-cfd by default
DATASET_PATH=${1:-"datasets"}
DOCKER_IMAGE=${2:-"junhal/openvocab-3d"}

USER=$(whoami)

# Mount the current path to /workspace
docker run \
    --gpus all \
    --shm-size=32g \
    -it \
    -v /home/$USER:/root \
    -v $(pwd):/workspace \
    -v $DATASET_PATH:/datasets \
    $DOCKER_IMAGE
