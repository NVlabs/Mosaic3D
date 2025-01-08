#!/bin/bash
#SBATCH --account=nvr_lpr_nvgptvision
#SBATCH --partition=grizzly,polar,polar2,polar3,polar4
#SBATCH --time=04:00:00                  # Adjust time limit as needed
#SBATCH --exclusive
#SBATCH --cpus-per-task=31
#SBATCH --mem=1843200
#SBATCH --output=slurm_outputs/%j.log
#SBATCH --job-name=openvocab-3d-training
#SBATCH --signal=SIGUSR1@300             # Send SIGUSR1 300 seconds before time limit
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8

# Usage:
#
# sbatch --nodes=2 ./scripts/train_multinode.sbatch [TAG] \
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

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME=^docker0,lo

# Training arguments
TRAINING_ARGS=("$@")

NODE=$(hostname -s)
USER=$(whoami)

# Project specific variables
ACCOUNT=nvr_lpr_nvgptvision
LUSTRE_NVR=/lustre/fsw/portfolios/nvr
LUSTRE_HOME="${LUSTRE_NVR}/users/${USER}"
PROJECT_ROOT="${LUSTRE_NVR}/projects/${ACCOUNT}"
DATA_DIR="${PROJECT_ROOT}/datasets"

# Image and code root
IMAGE_URL="gitlab-master.nvidia.com/3dmmllm/openvocab-3d:$TAG"
# Get the absolute current working directory
CODE_ROOT=$(pwd)

# HF_HOME for credentials and other Hugging Face data.
# HF_HUB_CACHE for caching repositories from the Hub.
# HF_ASSETS_CACHE for caching other assets.
# Setting environment variables
HF_HUB_CACHE="${PROJECT_ROOT}/huggingface_cache"

# Print tunneling instructions - only on rank 0
if [ "${SLURM_PROCID:-0}" -eq 0 ]; then
    echo -e "
    Running a multi-node GPU job on
        Node: ${NODE}
        JOB_ID ${SLURM_JOB_ID}
        Date: $(TZ=America/Los_Angeles date)
        User: ${USER}
        Image: ${IMAGE_URL}
        GPU_COUNT: ${GPU_COUNT}
        TRAINING_ARGS: ${TRAINING_ARGS[*]}
        Number of Nodes: ${SLURM_JOB_NUM_NODES}
    "
fi

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

TRAINER="ddp" # Default trainer for multi-node

# Your training script here - only print date and nvidia-smi on rank 0
CMD="
if [ \"\${SLURM_PROCID:-0}\" -eq 0 ]; then
    TZ=America/Los_Angeles date;
    nvidia-smi;
fi;
conda deactivate;
cd /workspace;
cd warpconvnet; pip install -e .; cd ..;
export HF_HUB_CACHE=${HF_HUB_CACHE};
export PYTHONPATH=/workspace;
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;
python3 -m \
    src.train \
    trainer=${TRAINER} \
    trainer.devices=${GPU_COUNT} \
    trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
    ${TRAINING_ARGS[*]}
"

set -x

# Pass CMD with double quotes
srun \
    --kill-on-bad-exit=1 \
    --container-image="$IMAGE_CACHE_FILE" \
    --container-mounts="$HOME:/root,/lustre:/lustre,${CODE_ROOT}:/workspace,${DATA_DIR}:/datasets" \
    bash -c "$CMD"

# Get the status from $OUTPUT_FOLDER/status.txt
STATUS_FILE="results/${SLURM_JOB_ID}/status.txt"
if [ -f "$STATUS_FILE" ]; then
    STATUS=$(cat "$STATUS_FILE")
    echo "Status: $STATUS"
    if [ "${STATUS}" == "STOPPED" ] || [ "${STATUS}" == "RUNNING" ]; then
        echo "Resubmitting the job with script: ${SCRIPT_PATH} ${*}"
        scontrol requeue "$SLURM_JOB_ID"
    fi
else
    echo "Status file not found: $STATUS_FILE"
fi
