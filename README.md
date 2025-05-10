<div align="center">

# Mosaic3D++: Open-Vocab 3D Understanding Foundation Models

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

</div>

This is a unpublished research codebase for Mosaic3D++: Open-Vocab 3D Understanding Foundation Models.
Do not distribute, modify, or share any part of the codebase without permission from the authors.
Please contact Chris Choy (cchoy@nvidia.com) if you want to use parts of this codebase.

## Description

This is a codebase for various open-vocab 3D understanding models, including open-set indoor semantic segmentations, etc.

## Installation

It is recommended to use [uv](https://github.com/astral-sh/uv) for managing Python dependencies. `uv` is an extremely fast Python package installer and resolver, written in Rust, and designed as a drop-in replacement for `pip` and `pip-tools`.

First, install `uv`:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, create a virtual environment and install the dependencies from `pyproject.toml`:

```bash
# Create a virtual environment (optional but recommended)
uv venv

# Activate the virtual environment
# On macOS and Linux
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"
```

Use of docker is still an option if preferred.

#### On Local Machines (Docker)

```bash
# build docker image. This command will automatically push the built image to GitLab registry
bash docker/docker_build.sh

# run docker container.
bash docker/docker_run.sh [path/to/datasets]
# e.g. bash docker/docker_run.sh /home/junhal/datasets
```

#### ORD (Docker)

On ORD cluster, all the installation and docker container-related stuffs are located in sbatch script: `script/train.sbatch`

Update appropriate variables in `script/train.sbatch` script.

## How to run

Train model with default configuration

```bash
# train RegionPLC model with default config (ScanNet-base15 config)
python src/train.py experiment=regionplc logger=wandb
```

If you run the experiments on ORD, use the following command for sbatch job submission

```bash
# scannet base 15
sbatch --gres=gpu:8 ./scripts/train.sbatch \
     experiment=regionplc \
     logger=auto_resume_wandb \
     seed=${SEED} \
     +trainer.precision="16-mixed"

# scannet base 12
sbatch --gres=gpu:8 ./scripts/train.sbatch \
     experiment=regionplc \
     data=regionplc_base12 \
     logger=auto_resume_wandb \
     +trainer.precision="16-mixed"

# scannet base 10
sbatch --gres=gpu:8 ./scripts/train.sbatch \
     experiment=regionplc \
     data=regionplc_base10 \
     logger=auto_resume_wandb \
     +trainer.precision="16-mixed"

# scannet zero-shot
sbatch --gres=gpu:8 ./scripts/train.sbatch \
     experiment=regionplc_openvocab \
     trainer.max_epochs=128 \
     logger=auto_resume_wandb \
     seed=${SEED} \
     +trainer.precision="16-mixed"

# Warp PointConv zero-shot
sbatch --gres=gpu:8 ./scripts/train.sbatch \
     experiment=regionplc_openvocab \
     model=warp_pointconv_enc_dec \
     trainer.max_epochs=128 \
     data.batch_size=2 \
     logger=auto_resume_wandb
```

## Development

Pre-commit is configured to ensure consistent code styling and check for syntax errors.

To install pre-commit hooks and ensure your merge requests pass these checks, run the following command:

```bash
pre-commit install
```

### Using the Latest WarpConvNet

The warpconvnet codebase is updated frequently. To use the latest version, clone the repository inside openvocab-3d repository and make sure the cluster job to use the cloned warpconvnet by installing the warpconvnet via pip directly.

1. Clone the warpconvnet repository inside openvocab-3d repository

```bash
cd openvocab-3d
git clone --recurse-submodules https://gitlab-master.nvidia.com/3dmmllm/warp.git warpconvnet
```

2. When running SLURM jobs, make sure to add `pip install --force-reinstall --no-deps ./warpconvnet` before running the training script.

```bash
# Your train.sbatch
CMD="
...
pip install --force-reinstall --no-deps ./warpconvnet
torchrun ...
"
```

## Debugging Tips

Use the following command to enable CUDA kernel launch blocking.
For pytorch unique CUDA illegal memory access error, use the warp docker image tag `warp` on `gitlab-master.nvidia.com/3dmmllm/openvocab-3d:warp`.
