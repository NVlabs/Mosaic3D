#!/bin/bash
#SBATCH --account=nvr_lpr_nvgptvision
#SBATCH --partition=batch_singlenode,grizzly,polar,polar2,polar3,polar4
#SBATCH --time=04:00:00                  # Adjust time limit as needed
#SBATCH --overcommit                     # allows more than one process per CPU
#SBATCH --cpus-per-task=16               # number of cores
#SBATCH --mem-per-cpu=64G                # memory per CPU core
#SBATCH --output=slurm_outputs/%j.log
#SBATCH --job-name=openvocab-3d-training
#SBATCH --signal=SIGUSR1@300             # Send SIGUSR1 300 seconds before time limit

# Usage:
#
# sbatch --gres=gpu:8 ./scripts/train.sbatch \
#     experiment=train_lseg3d \
#     trainer=ddp \
#     trainer.devices=8 \
#     data.train_dataset.data_dir=${DATASET_ROOT}/SceneVerse \
#     data.val_dataset.datapath_prefix=${DATASET_ROOT}/OpenScene/scannet_3d

# Get the number of gpus from scontrol and strip all the white spaces and new lines
# example output
# >    TresPerNode=gres:gpu:2
GPU_COUNT=$(scontrol show job $SLURM_JOBID | grep gres | awk -F: '{print $3}' | tr -d '[:space:]')

# Training arguments
TRAINING_ARGS=("$@")

NODE=$(hostname -s)
USER=$(whoami)

# Project specific variables
ACCOUNT=nvr_lpr_nvgptvision
LUSTRE_NVR=/lustre/fsw/portfolios/nvr
LUSTRE_HOME="${LUSTRE_NVR}/users/${USER}"
PROJECT_ROOT="${LUSTRE_NVR}/projects/${ACCOUNT}"

# Image and code root
IMAGE_URL="gitlab-master.nvidia.com/3dmmllm/openvocab-3d"
CODE_ROOT="${LUSTRE_HOME}/projects/openvocab-3d"
CURR_SCRIPT="${CODE_ROOT}/scripts/train.sbatch"

# HF_HOME for credentials and other Hugging Face data.
# HF_HUB_CACHE for caching repositories from the Hub.
# HF_ASSETS_CACHE for caching other assets.
# Setting environment variables
HF_HUB_CACHE="${PROJECT_ROOT}/huggingface_cache"

# Print tunneling instructions
echo -e "
Running a GPU job on

    Node: ${NODE}
    JOB_ID ${SLURM_JOB_ID}
    Date: $(TZ=America/Los_Angeles date)
    User: ${USER}
    Image: ${IMAGE_URL}
    GPU_COUNT: ${GPU_COUNT}
    TRAINING_ARGS: ${TRAINING_ARGS[@]}
"

# Cache docker file
# Check if the cache_image.sh exists
if [ -f $LUSTRE_HOME/sbatch/cache_image.sh ]; then
    # Cache the image
    source $LUSTRE_HOME/sbatch/cache_image.sh

    SQSH_CACHE_DIR=${PROJECT_ROOT}/enroot-cache
    IMAGE_CACHE_FILE=$(cache_image $IMAGE_URL $ACCOUNT $SQSH_CACHE_DIR)
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
torchrun \
    --nnodes=1 \
    --nproc_per_node=${GPU_COUNT} \
    src/train.py \
    ${TRAINING_ARGS[@]}
"

set -x
# Pass CMD with double quotes
srun \
    --container-image=$IMAGE_CACHE_FILE \
    --container-mounts="$HOME:/root,/lustre:/lustre,${CODE_ROOT}:/workspace" \
    bash -c "$CMD"

# Get the status from $OUTPUT_FOLDER/status.txt
STATUS_FILE="results/${SLURM_JOB_ID}/status.txt"
if [ -f $STATUS_FILE ]; then
    STATUS=$(cat $STATUS_FILE)
    echo "Status: $STATUS"
    if [ "${STATUS}" == "STOPPED" ] || [ "${STATUS}" == "RUNNING" ]; then
        echo "Resubmitting the job with script: ${SCRIPT_PATH} ${@}"
        scontrol requeue $SLURM_JOB_ID
    fi
else
    echo "Status file not found: $STATUS_FILE"
fi

