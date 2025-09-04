<div align="center">

# Mosaic3D: Foundation Dataset and Model for Open-vocabulary 3D Segmentation (CVPR 2025)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://nvlabs.github.io/Mosaic3D/)
[![Paper](https://img.shields.io/badge/CVPR-2025-green)](https://arxiv.org/abs/YOUR_ARXIV_ID)

**[Junha Lee¹'²*](https://junha-l.github.io/), [Chunghyun Park¹'²*](https://chrockey.github.io/), [Jaesung Choe¹](https://jaesung-choe.github.io/), [Frank Wang¹](https://vllab.ee.ntu.edu.tw/ycwang.html), [Jan Kautz¹](https://research.nvidia.com/person/jan-kautz), [Minsu Cho²](https://cvlab.postech.ac.kr/~mcho/), [Chris Choy¹](https://chrischoy.github.io/)**

*equal contribution\
¹NVIDIA, ²POSTECH

</div>

## Overview

We present **Mosaic3D**, a comprehensive solution for open-vocabulary 3D scene understanding that addresses three essential aspects: precise 3D region segmentation, comprehensive textual descriptions, and sufficient dataset scale. Our approach combines state-of-the-art open-vocabulary image segmentation models with region-aware vision-language models to create an automatic pipeline for generating high-quality 3D mask-text pairs.

### Key Contributions

- **Mosaic3D-5.6M Dataset**: The largest 3D mask-text paired dataset to date, encompassing over 30K indoor scenes and approximately 1M RGB-D frames, yielding 5.6M region captions with 30M total text tokens
- **Mosaic3D Model**: A 3D visual foundation model (3D-VFM) combining a 3D encoder trained with contrastive learning and a lightweight mask decoder for open-vocabulary 3D semantic and instance segmentation
- **State-of-the-art Performance**: Achieves leading results on open-vocabulary 3D semantic and instance segmentation benchmarks including ScanNet200, Matterport3D, and ScanNet++

### Dataset Advantages

Our Mosaic3D-5.6M dataset offers significant advantages over existing datasets:

- **Scale**: 5.6M mask-text pairs across 30K+ scenes (significantly larger than existing datasets)
- **Precision**: Leverages advanced open-vocabulary segmentation for precise region boundaries
- **Rich Descriptions**: Captures object attributes, spatial relationships, and scene context
- **Quality**: Combines robust region-aware VLMs for comprehensive textual annotations

## Dataset

### Mosaic3D-5.6M Download

The dataset can be found in [Huggingface](https://huggingface.co/datasets/junhalee/Mosaic3D). Follow the instruction there to download and organize the data into required structure.


## Environment Setup

### Docker (Recommended)

```bash
# Build docker image
bash docker/docker_build.sh

# Run docker container with dataset path
bash docker/docker_run.sh /path/to/datasets
```

### Conda Environment

```bash
# Create conda environment
conda env create -f environment.yaml

# Install requirements
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install
```

## Model Architecture

Mosaic3D employs a two-stage training approach:

1. **Per-point Language Alignment**: Trains a 3D encoder using contrastive learning to align 3D point features with textual descriptions
2. **Mask Decoder Training**: Trains a lightweight mask decoder to predict instance segments from the aligned features

This design enables effective open-vocabulary 3D semantic and instance segmentation across diverse indoor scenes.

## Training

### Encoder Training

```bash
# Train Mosaic3D model with default configuration
python src/train.py experiment=train_spunet_scannet trainer.ddp trainer.devices=8 logger=wandb

# Evaluate trained model
python src/eval.py ckpt_path=/path/to/checkpoint.ckpt data=scannet
```

### Mask Decoder Training

```bash
# Download Segment3D checkpoint
python src/models/networks/opensegment3d/download_ckpt.py

# Train a lightweight mask decoder with default configuration
python src/train.py experiment=train_opensegment3d_scannet model.net.backbone_ckpt=/path/to/encoder.ckpt trainer.ddp trainer.devices=8 logger=wandb
```

### Configuration Override

You can override any configuration parameter from the command line:

```bash
python src/train.py experiment=train_spunet_scannet data=sc model=spunet34c trainer.max_epochs=100
```

## Evaluation

The model achieves state-of-the-art results on multiple benchmarks:

- **Annotation-free 3D semantic segmentation**: ScanNet, Matterport3D, ScanNet++
- **Annotation-free 3D instance segmentation**: ScanNet200

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{lee2025mosaic3d,
  title={Mosaic3d: Foundation dataset and model for open-vocabulary 3d segmentation},
  author={Lee, Junha and Park, Chunghyun and Choe, Jaesung and Wang, Yu-Chiang Frank and Kautz, Jan and Cho, Minsu and Choy, Chris},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={14089--14101},
  year={2025}
}
```


## Acknowledgments

Our work builds upon several fantastic open-source projects. We'd like to express our gratitude to the authors of:
- [Pointcept](https://github.com/Pointcept/Pointcept)
- [PLA & RegionPLC](https://github.com/CVMI-Lab/PLA)
- [SPConv](https://github.com/traveller59/spconv)
- [Segment3D](https://github.com/LeapLabTHU/Segment3D)
