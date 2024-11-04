import fire
import hydra
import yaml
from omegaconf import OmegaConf
import warp as wp
from tqdm import tqdm
from src.data.caption_transform import CaptionFilter
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List, Dict, Tuple
from collections import defaultdict


def process_batch(
    batch_indices: List[int], dataset, caption_filter: CaptionFilter
) -> Tuple[Dict[str, int], List[Tuple[str, List[str]]]]:
    """Process a batch of dataset indices in parallel.

    Args:
        batch_indices: List of dataset indices to process
        dataset: The dataset instance
        caption_filter: CaptionFilter instance

    Returns:
        Tuple of (stats_dict, filtered_examples)
    """
    stats = {
        "total_captions": 0,
        "valid_captions": 0,
    }
    deleted_captions = defaultdict(list)

    for idx in batch_indices:
        data = dataset[idx]
        if "caption_data" in data and "caption" in data["caption_data"]:
            captions = data["caption_data"]["caption"]
            scene_name = data.get("scene_name", f"scene_{idx}")

            # Handle both single and multiple captions
            if isinstance(captions, str):
                captions = [captions]

            stats["total_captions"] += len(captions)

            # Apply filtering
            valid_flags = caption_filter(captions)
            stats["valid_captions"] += sum(valid_flags)

            # Store a few examples of filtered captions with their failure reasons
            for caption, is_valid in zip(captions, valid_flags):
                if not is_valid:
                    deleted_captions[scene_name].append(caption)
                    print(f"Removed caption: {caption}")

    return stats, deleted_captions


def analyze_filtered_captions(
    base_config: str = "configs/data/scannet.yaml",
    config: str = "configs/data/scannet_125k.yaml",
    transform: str = "configs/data/transforms/ptv3_transforms_caption.yaml",
    num_samples: int = 5,
    num_workers: int = None,
) -> None:
    """Analyze how many captions are filtered by the transforms."""
    # Load and merge configs
    with open(base_config) as f:
        base_config = yaml.safe_load(f.read())

    with open(config) as f:
        config_dict = yaml.safe_load(f.read())

    with open(transform) as f:
        transform_config = yaml.safe_load(f.read())

    # Create merged config
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

    if "defaults" in config_dict:
        del config_dict["defaults"]
    merged_config["data"]["train_dataset"].update(config_dict["train_dataset"])

    # Create OmegaConf and instantiate dataset
    cfg = OmegaConf.create(merged_config)
    dataset = hydra.utils.instantiate(cfg.data.train_dataset)()

    # Initialize caption filter
    caption_filter = CaptionFilter(
        min_words=1,
        max_words=50,
        min_letter_ratio=0.5,
        max_repetition_ratio=0.4,
        max_consecutive=3,
    )

    # Setup parallel processing
    if num_workers is None:
        num_workers = max(1, 16)

    # Split dataset into batches
    total_indices = list(range(len(dataset)))
    batch_size = len(total_indices) // num_workers
    batches = [total_indices[i : i + batch_size] for i in range(0, len(total_indices), batch_size)]

    # Process batches in parallel
    print(f"Processing {len(dataset)} samples with {num_workers} workers...")
    process_fn = partial(process_batch, dataset=dataset, caption_filter=caption_filter)

    total_stats = defaultdict(int)
    removed_captions = defaultdict(list)

    with Pool(num_workers) as pool:
        for batch_stats, batch_examples in tqdm(
            pool.imap(process_fn, batches), total=len(batches)
        ):
            # Accumulate statistics
            for k, v in batch_stats.items():
                total_stats[k] += v

            # Collect removed captions
            removed_captions.update(batch_examples)

    # Calculate final statistics
    total_captions = total_stats["total_captions"]
    valid_captions = total_stats["valid_captions"]
    filtered_out = total_captions - valid_captions
    filtered_percentage = (filtered_out / total_captions * 100) if total_captions > 0 else 0

    # Print results
    print("\nCaption Filtering Analysis:")
    print(f"Total captions analyzed: {total_captions}")
    print(f"Captions that passed filters: {valid_captions}")
    print(f"Captions filtered out: {filtered_out}")
    print(f"Percentage filtered: {filtered_percentage:.2f}%")

    # Print filter criteria
    print("\nFilter Criteria Used:")
    print(f"- Minimum words: {caption_filter.min_words}")
    print(f"- Maximum words: {caption_filter.max_words}")
    print(f"- Minimum letter ratio: {caption_filter.min_letter_ratio * 100}%")
    print(f"- Maximum word repetition ratio: {caption_filter.max_repetition_ratio * 100}%")
    print(f"- Maximum consecutive repeats: {caption_filter.max_consecutive}")

    # Show examples of filtered captions
    if removed_captions:
        print("\nExample Captions That Were Filtered Out:")
        for i, (caption, failed_filters) in enumerate(removed_captions[:num_samples], 1):
            print(f"\n{i}. Caption: {caption}")
            print(f"   Failed filters: {', '.join(failed_filters)}")


if __name__ == "__main__":
    wp.init()
    fire.Fire(analyze_filtered_captions)
