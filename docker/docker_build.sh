#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS] [VERSION]"
    echo "Options:"
    echo "  -h, --help    Display this help message"
    echo "  -t, --tag     Tag the Docker image with the provided version"
    echo "  -n, --no-cache Build the Docker image without using cache"
    echo "VERSION defaults to 'latest' if not provided"
}

# Initialize variables
VERSION="latest"
TAG="mosaic3d:${VERSION}"
NO_CACHE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h | --help)
            usage
            exit 0
            ;;
        -t | --tag)
            TAG=$2
            shift 2
            ;;
        -n | --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        *)
            VERSION=$1
            shift
            ;;
    esac
done

# Build the Docker image
# Add --no-cache to force a rebuild
echo "Building Docker image..."
docker build \
    $NO_CACHE \
    -t "$TAG" \
    -f docker/Dockerfile .

# Test docker
if ! docker run --gpus all -it --rm "$TAG" python -c "import torch, MinkowskiEngine, pointnet2, pointops; print(f'CUDA available: {torch.cuda.is_available()}, MinkowskiEngine: {MinkowskiEngine.__version__}, pointnet2 and pointops imported successfully')"; then
    echo "Docker test failed"
    exit 1
fi

echo "Docker image built and tested successfully"
echo "To push the image, run the script with the -p or --push option"
