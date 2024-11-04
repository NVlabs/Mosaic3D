from typing import List

import pytest
import hydra
import torch
import yaml
from omegaconf import OmegaConf
from lightning import LightningDataModule
import warp as wp
from src.data.caption_transform import CaptionFilter


def test_caption_filter():
    """Test the CaptionFilter class with various caption scenarios."""
    # Initialize filter with default parameters
    caption_filter = CaptionFilter(
        min_words=3,
        max_words=50,
        min_letter_ratio=0.5,
        max_repetition_ratio=0.4,
        max_consecutive=3,
    )

    # Test cases
    test_cases = [
        # Valid captions
        ("A red chair in the corner", True),
        ("The large window lets in natural light", True),
        # Invalid cases
        ("", False),  # Empty caption
        ("a b", False),  # Too few words
        ("123 456 789", False),  # No letters
        ("the the the the desk", False),  # Too much repetition
        ("chair chair chair chair", False),  # Consecutive repeats
        ("x " * 51, False),  # Too many words
    ]

    for caption, expected in test_cases:
        result = caption_filter([caption])[0]
        assert result == expected, f"Failed for caption: '{caption}'"


def test_caption_filter_batch():
    """Test the CaptionFilter class with batch processing."""
    caption_filter = CaptionFilter()

    captions = [
        "A red chair in the corner",
        "",  # Empty caption
        "The large window lets in natural light",
        "a b",  # Too few words
    ]

    expected = [True, False, True, False]
    result = caption_filter(captions)

    assert result == expected


def test_debug_caption():
    """Test the debug_caption method for detailed filter results."""
    caption_filter = CaptionFilter()

    caption = "The large window lets in natural light"
    debug_info = caption_filter.debug_caption(caption)

    # Check that all filter results are present
    expected_filters = [
        "FilterCaptionEmpty",
        "FilterCaptionWordCount",
        "FilterCaptionLetterRatio",
        "FilterCaptionWordRepetition",
        "FilterCaptionConsecutiveRepeats",
        "FilterCaptionPhraseRepeats",
    ]

    for filter_name in expected_filters:
        assert filter_name in debug_info
        assert isinstance(debug_info[filter_name], bool)


def test_filter_captions_with_scenes():
    """Test filtering captions with corresponding scene names."""
    caption_filter = CaptionFilter()

    captions = [
        "A red chair in the corner",
        "",  # Empty caption
        "The large window lets in natural light",
        "a b",  # Too few words
    ]

    scene_names = ["scene_0001", "scene_0002", "scene_0003", "scene_0004"]

    filtered_captions, filtered_scenes = caption_filter.filter_captions(captions, scene_names)

    assert len(filtered_captions) == len(filtered_scenes)
    assert len(filtered_captions) == 2  # Only valid captions should remain
    assert filtered_scenes == ["scene_0001", "scene_0003"]


def test_scannet_caption_loading():
    """Test loading ScanNet data with caption transforms."""
    # Load the base scannet config first
    with open("configs/data/scannet.yaml") as f:
        base_config = yaml.safe_load(f.read())

    # Load the scannet_125k config
    with open("configs/data/scannet_125k.yaml") as f:
        config_dict = yaml.safe_load(f.read())

    # Load the transform config
    with open("configs/data/transforms/ptv3_transforms_caption.yaml") as f:
        transform_config = yaml.safe_load(f.read())

    # Create base config structure
    merged_config = {
        "data": {
            "_target_": base_config["_target_"],
            "train_dataset": base_config["train_dataset"],
            "val_datasets": base_config["val_datasets"],
            "train_transforms": transform_config["train_transforms"],
            "val_transforms": transform_config["val_transforms"],
            "collate_fn": base_config["collate_fn"],
            "batch_size": base_config["batch_size"],
            "num_workers": 0,
            "checkpoint_monitor": base_config["checkpoint_monitor"],
        }
    }

    # Update with scannet_125k specific configs
    if "defaults" in config_dict:
        del config_dict["defaults"]
    merged_config["data"]["train_dataset"].update(config_dict["train_dataset"])

    # Create OmegaConf
    cfg = OmegaConf.create(merged_config)

    # Instantiate datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")

    # Get train dataloader
    train_loader = datamodule.train_dataloader()

    # Test a few batches
    for batch_idx, batch in enumerate(train_loader):
        # Check basic batch structure
        assert isinstance(batch, dict)
        assert "coord" in batch
        assert "color" in batch
        assert "caption_data" in batch

        # Check caption data structure
        caption_data = batch["caption_data"]
        assert "caption" in caption_data
        assert "idx" in caption_data

        # Check that captions are properly filtered
        batch_captions: List[List[str]] = caption_data["caption"]
        for captions in batch_captions:
            # Check minimum word count
            for caption in captions:
                assert len(caption.split()) >= 1, f"Caption '{caption}' has too few words"

                # Check letter ratio
                letter_count = sum(char.isalpha() for char in caption)
                total_count = sum(not char.isspace() for char in caption)
                assert letter_count / total_count >= 0.5

        # Check tensor dimensions
        assert batch["coord"].dim() == 2  # [N, 3]
        assert batch["color"].dim() == 2  # [N, 3]

        # Only test first 2 batches
        if batch_idx >= 1:
            break


def test_caption_filter_integration():
    """Test that caption filtering is properly integrated in the data pipeline."""
    # Initialize caption filter with config settings
    caption_filter = CaptionFilter(
        min_words=3,
        max_words=50,
        min_letter_ratio=0.5,
        max_repetition_ratio=0.4,
        max_consecutive=3,
    )

    # Test sample captions that should be filtered
    test_captions = [
        "A red chair in the corner of the room",  # Valid
        "a b",  # Too short
        "The chair chair chair chair",  # Too repetitive
        "123 456 789",  # No letters
        "This is a valid caption with good content",  # Valid
    ]

    results = caption_filter(test_captions)
    assert results == [True, False, False, False, True]


if __name__ == "__main__":
    wp.init()
    pytest.main([__file__, "-v"])
