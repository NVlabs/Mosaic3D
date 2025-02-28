#!/bin/bash
#SBATCH --partition=A100-80GB
#SBATCH -q hpgpu
#SBATCH --time=3-00:00:00                  # Adjust time limit as needed
#SBATCH --cpus-per-task=32               # number of cores
#SBATCH --output=slurm_outputs/%j.log
#SBATCH --job-name=mosaic3d-training
#SBATCH --signal=SIGUSR1@300             # Send SIGUSR1 300 seconds before time limit
#SBATCH --gres=gpu:8

# Usage:
#
# sbatch --gres=gpu:8 ./scripts/train_gsai.sbatch [TAG] \
#      experiment=regionplc_openvocab \
#      trainer=ddp \
#      logger=auto_resume_wandb

# Get the number of gpus from scontrol and strip all the white spaces and new lines
# example output
# >    TresPerNode=gres:gpu:2
GPU_COUNT=$(scontrol show job "$SLURM_JOBID" | grep gres | awk -F: '{print $3}' | tr -d '[:space:]')

# Set default TAG to "warp"
TAG="latest"

# Check if the next argument doesn't start with "--" or contains "=" (indicating it's a TAG)
if [[ $# -gt 0 && ${1:0:2} != "--" && ! $1 =~ "=" ]]; then
    TAG=$1
    shift
fi

# Training arguments
TRAINING_ARGS=("$@")

NODE=$(hostname -s)
USER=$(whoami)

# Project specific variables
USER_ROOT="/home/${USER}"
DATA_DIR="${USER_ROOT}/datasets"

# Image and code root
IMAGE_URL="chrockey/mosaic3d:$TAG"
# Get the absolute current working directory
CODE_ROOT=$(pwd)

# HF_HOME for credentials and other Hugging Face data.
# HF_HUB_CACHE for caching repositories from the Hub.
# HF_ASSETS_CACHE for caching other assets.
# Setting environment variables
HF_HUB_CACHE="${USER_ROOT}/huggingface_cache"

# Print tunneling instructions
echo -e "
Running a GPU job on

    Node: ${NODE}
    JOB_ID ${SLURM_JOBID}
    CKPT_JOB_ID ${CKPT_SLURM_JOBID:-${SLURM_JOBID}}
    Date: $(TZ=Asia/Seoul date)
    User: ${USER}
    Image: ${IMAGE_URL}
    GPU_COUNT: ${GPU_COUNT}
    TRAINING_ARGS: ${TRAINING_ARGS[*]}
"

TRAINER="ddp" # Default trainer
# If GPU_COUNT==1, set trainer to 'gpu'
if [ "$GPU_COUNT" -eq 1 ]; then
    TRAINER="gpu"
fi

# Your training script here
CMD="
TZ=Asia/Seoul date;
cd /workspace;
conda deactivate;
nvidia-smi;
export HF_HUB_CACHE=${HF_HUB_CACHE};
export PYTHONPATH=/workspace;
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;
pip install flash-attn;
pip install -U lightning;
torchrun \
    --nnodes=1 \
    --nproc_per_node=${GPU_COUNT} \
    src/train.py \
    trainer=${TRAINER} \
    trainer.devices=${GPU_COUNT} \
    ${TRAINING_ARGS[*]}
"

# Docker
DOCKER_IMAGE=${IMAGE_URL}
CONTAINER_NAME="${USER}_mosaic3d_${SLURM_JOBID}"

# UUID
UUIDLIST=$(nvidia-smi -L | cut -d '(' -f 2 | awk '{print$2}' | tr -d ")" | paste -s -d, -)
GPULIST=\"device=${UUIDLIST}\"
echo "Using GPUs with UUID: ${UUIDLIST}"

set -x
# Pass CMD with double quotes
docker run \
    --rm \
    --name "${CONTAINER_NAME}" \
    --gpus ${GPULIST} \
    -v "$HOME:/root" \
    -v "${CODE_ROOT}:/workspace" \
    -v "${DATA_DIR}:/datasets" \
    ${DOCKER_IMAGE} \
    bash -c "$CMD"

# Get the status from $OUTPUT_FOLDER/status.txt
STATUS_FILE="results/${CKPT_SLURM_JOBID:-${SLURM_JOBID}}/status.txt"
if [ -f "$STATUS_FILE" ]; then
    STATUS=$(cat "$STATUS_FILE")
    echo "Status: $STATUS"
    if [ "${STATUS}" == "STOPPED" ] || [ "${STATUS}" == "RUNNING" ]; then
        echo "Resubmitting the job with script: ${SCRIPT_PATH} ${*}"
        scontrol requeue "$SLURM_JOBID"
    fi
else
    echo "Status file not found: $STATUS_FILE"
fi
