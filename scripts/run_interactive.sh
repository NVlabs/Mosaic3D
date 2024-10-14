#!/bin/bash

# Usage: bash ./scripts/run_interactive.sh [TAG] [PORT]

# Project specific variables
ACCOUNT=nvr_lpr_nvgptvision
LUSTRE_NVR=/lustre/fsw/portfolios/nvr
LUSTRE_HOME="${LUSTRE_NVR}/users/${USER}"
PROJECT_ROOT="${LUSTRE_NVR}/projects/${ACCOUNT}"
DATA_DIR="${PROJECT_ROOT}/datasets"

# Image and code root
TAG=${1:-"warp"}
IMAGE_URL="gitlab-master.nvidia.com/3dmmllm/openvocab-3d:$TAG"
# Get the absolute current working directory
CODE_ROOT=$(pwd)
PORT=${2:-22000}

# HF_HOME for credentials and other Hugging Face data.
# HF_HUB_CACHE for caching repositories from the Hub.
# HF_ASSETS_CACHE for caching other assets.
# Setting environment variables
# Use user's directory due to limited space in project directory
HF_HUB_CACHE="${LUSTRE_HOME}/huggingface/cache"

# Print tunneling instructions
echo -e "
Running an interactive job on

    Date: $(TZ=America/Los_Angeles date)
    Image: ${IMAGE_URL}
"

# Cache docker file
# Check if the cache_image.sh exists
if [ -f "$LUSTRE_HOME/sbatch/cache_image.sh" ]; then
    # Cache the image
    # shellcheck source=/dev/null
    source "$LUSTRE_HOME/sbatch/cache_image.sh"

    SQSH_CACHE_DIR=${PROJECT_ROOT}/enroot-cache
    IMAGE_CACHE_FILE=$(cache_image "$IMAGE_URL" "$ACCOUNT" "$SQSH_CACHE_DIR")
else
    echo "$LUSTRE_HOME/sbatch/cache_image.sh does not exist. Using ${IMAGE_URL} without caching."
    IMAGE_CACHE_FILE=${IMAGE_URL}
fi

# Your training script here
CMD="
TZ=America/Los_Angeles date;
cd /workspace;
conda deactivate;
nvidia-smi;
export HF_HUB_CACHE=${HF_HUB_CACHE};
export PYTHONPATH=/workspace;
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;
bash $(pwd)/scripts/setup_interactive.sh $PORT
"

set -x
# Pass CMD with double quotes
srun \
    --account="${ACCOUNT}" \
    --partition="interactive" \
    --no-container-remap-root \
    --job-name="interactive" \
    --gpus=8 \
    --exclusive \
    --time=04:00:00 \
    --container-image="$IMAGE_CACHE_FILE" \
    --container-mounts="$HOME:/root,/lustre:/lustre,${CODE_ROOT}:/workspace,${DATA_DIR}:/datasets" \
    --pty /bin/bash -c "$CMD"
