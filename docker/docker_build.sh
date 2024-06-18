#!/bin/bash

TAG=${1:-"junhal/openvocab-3d"}
IMAGE_URL=${2:-"gitlab-master.nvidia.com/3dmmllm/openvocab-3d"}

# Build the Docker image
# Add --no-cache to force a rebuild
docker build \
    -t $TAG \
    -t $IMAGE_URL \
    -f docker/Dockerfile .

# Exit if previous build failed
if [ $? -ne 0 ]; then
    echo "Docker build failed"
    exit 1
fi

# Test docker
docker \
    run \
    --gpus all \
    -it --rm \
    $TAG \
    python -c "import torch;print(torch.cuda.is_available());"

# Get the exit code and return if failed
if [ $? -ne 0 ]; then
    echo "Docker test failed"
    exit 1
fi

docker push $IMAGE_URL