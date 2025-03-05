#!/bin/bash
#SBATCH --partition=A6000,RTX6000ADA,L40S
#SBATCH --time=3-00:00:00                  # Adjust time limit as needed
#SBATCH --cpus-per-task=32               # number of cores
#SBATCH --mem=250000
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
# NOTE: For A100, add -q hpgpu

# Get the number of gpus from scontrol and strip all the white spaces and new lines
# example output
# >    TresPerNode=gres:gpu:2
GPU_COUNT=$(scontrol show job "$SLURM_JOBID" | grep "TresPerNode" | grep -o "gres/gpu:[0-9]*" | cut -d: -f2 | tr -d '[:space:]')

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
DATA_DIR="$HOME/datasets"

# Image and code root
DOCKER_IMAGE="chrockey/mosaic3d:$TAG"
# Get the absolute current working directory
CODE_ROOT=$(pwd)

# Setting environment variables

# Print tunneling instructions
echo -e "
Running a GPU job on

    Node: ${NODE}
    JOB_ID ${SLURM_JOBID}
    CKPT_JOB_ID ${CKPT_SLURM_JOBID:-${SLURM_JOBID}}
    Date: $(TZ=Asia/Seoul date)
    User: ${USER}
    Image: ${DOCKER_IMAGE}
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

git config --global --add safe.directory /workspace;

export PYTHONPATH=/workspace;
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;
export SLURM_JOBID=${SLURM_JOBID};
export CKPT_SLURM_JOBID=${CKPT_SLURM_JOBID:-${SLURM_JOBID}};
pip install flash-attn;
pip install -U lightning 'open_clip_torch>=2.31.0' 'timm>=1.0.15';
torchrun \
    --nnodes=1 \
    --nproc_per_node=${GPU_COUNT} \
    src/train.py \
    trainer=${TRAINER} \
    trainer.devices=${GPU_COUNT} \
    ${TRAINING_ARGS[*]}
"

# Docker
CONTAINER_NAME="${USER}_mosaic3d_${SLURM_JOBID}"

# UUID
UUIDLIST=$(nvidia-smi -L | cut -d '(' -f 2 | awk '{print$2}' | tr -d ")" | paste -s -d, -)
GPULIST=\"device=${UUIDLIST}\"
echo "Using GPUs with UUID: ${UUIDLIST}"

# Define function to stop and remove containers
stop_and_remove_container() {
    local container_name=$1
    if docker ps -q --filter "name=${container_name}" | grep -q .; then
        echo "Stopping and removing container ${container_name}..."
        docker stop "${container_name}" && docker rm "${container_name}"
        echo "Container ${container_name} removed."
    fi
}

# Clean up existing containers with username prefix
echo "Checking for existing containers with prefix ${USER}_..."
EXISTING_CONTAINERS=$(docker ps -q --filter "name=${USER}_")
if [ -n "$EXISTING_CONTAINERS" ]; then
    echo "Found existing containers. Stopping and removing them..."
    for container in $EXISTING_CONTAINERS; do
        container_name=$(docker ps --format "{{.Names}}" -f "id=$container")
        stop_and_remove_container "$container_name"
    done
    echo "Existing containers removed."
else
    echo "No existing containers found with prefix ${USER}_."
fi

set -x
# Run the container
docker run \
    --rm \
    --shm-size=32G \
    --name "${CONTAINER_NAME}" \
    --gpus ${GPULIST} \
    -v "$HOME:/root" \
    -v "${CODE_ROOT}:/workspace" \
    -v "${DATA_DIR}:/datasets" \
    ${DOCKER_IMAGE} \
    bash -c "$CMD"

# Ensure container is removed if still running
stop_and_remove_container "${CONTAINER_NAME}"

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
