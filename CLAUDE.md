# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is Mosaic3D, an open-vocabulary 3D understanding research codebase from NVIDIA. It supports training and evaluation of various 3D neural network models for tasks like indoor semantic segmentation, focusing on open-vocabulary capabilities. The project combines multiple datasets (ScanNet, ARKitScenes, Matterport3D, etc.) and models (RegionPLC, Segment3D, OpenSegment3D, etc.).

## Core Architecture

The codebase uses **PyTorch Lightning** with **Hydra** for configuration management. Key architectural components:

- **Data Pipeline**: Multi-dataset support with unified transforms and collation (`src/data/`)
- **Model Framework**: Lightning modules with modular network architectures (`src/models/`)
- **Configuration System**: Hydra-based configs for reproducible experiments (`configs/`)
- **Training Pipeline**: Distributed training with SLURM cluster support (`scripts/`)

### Key Directories

- `src/data/`: Dataset implementations, transforms, and data loading
- `src/models/`: Neural network architectures and Lightning modules
- `configs/`: Hydra configuration files for experiments, models, and data
- `scripts/`: SLURM batch scripts for cluster training
- `warpconvnet/`: Submodule for WarpConvNet sparse convolution library

## Development Commands

### Environment Setup
```bash
# Docker-based development (recommended)
bash docker/docker_build.sh
bash docker/docker_run.sh /path/to/datasets

# Or conda environment
conda env create -f environment.yaml
pip install -r requirements.txt
```

### Training Commands
```bash
# Basic training with RegionPLC
python src/train.py experiment=regionplc logger=wandb

# Training on SLURM cluster
sbatch --gres=gpu:8 ./scripts/train.sbatch \
    experiment=regionplc \
    logger=auto_resume_wandb \
    +trainer.precision="16-mixed"
```

### Evaluation
```bash
# Evaluate checkpoint
python src/eval.py ckpt_path=/path/to/checkpoint.ckpt data=scannet
```

### Code Quality
```bash
# Install pre-commit hooks
pre-commit install

# Run formatting and linting
pre-commit run --all-files

# Run tests
pytest tests/
```

## Configuration System

The project uses Hydra with a composable configuration system:

- **experiments/**: Complete experiment configurations
- **data/**: Dataset and transform configurations
- **model/**: Model architecture and loss configurations
- **trainer/**: Training setup (GPU, distributed, etc.)
- **logger/**: Logging configurations (WandB, TensorBoard, etc.)

Override any config from command line:
```bash
python src/train.py data=scannet model=spunet34c trainer.max_epochs=100
```

## Dataset Management

Download RegionPLC dataset (update FedAuth token first):
```bash
python -m src.data.regionplc.download --download_dir /path/to/save/dataset
```

Supported datasets: ScanNet, ARKitScenes, Matterport3D, ScanNet++, Structured3D, EmbodiedScan, MMScan, Sceneverse.

## Model Architectures

Key models implemented:
- **RegionPLC**: Region-based point-language-color alignment
- **Segment3D**: 3D instance segmentation with language
- **OpenSegment3D**: Open-vocabulary 3D segmentation
- **WarpConvNet**: Sparse convolution networks via submodule

Each model has corresponding Lightning modules in `src/models/lightning_modules/`.

## HPC/Cluster Usage

For NVIDIA cluster training:
- Update variables in `scripts/train.sbatch`
- Use multi-node training with `scripts/train_multinode.sbatch`
- Results saved to `results/{SLURM_JOB_ID}/`
- Logs in `slurm_outputs/`

## Important Dependencies

- PyTorch 2.2.2 with Lightning 2.2
- spconv-cu120 for sparse convolutions
- Hydra 1.3.2 for configuration
- open_clip_torch for CLIP models
- transformers for language models
- Various 3D processing: open3d, plyfile, scipy

## Common Issues

- **CUDA Memory**: Use `+trainer.precision="16-mixed"` for mixed precision
- **WarpConvNet**: May need manual installation: `pip install --force-reinstall --no-deps ./warpconvnet`
- **Distributed Training**: Ensure proper GPU count matches trainer config
