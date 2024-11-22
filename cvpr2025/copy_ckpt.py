import concurrent.futures
import os
import subprocess
from pathlib import Path

import yaml
from rich.console import Console

console = Console()


def sync_single_directory(source_dir, target_dir):
    """Sync a single directory pair using rsync."""
    cmd = [
        "rsync",
        "-av",
        "--info=progress2",
        "--ignore-existing",
        "--exclude",
        "*wandb*",
        f"{source_dir}/",
        f"{target_dir}/",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "source": source_dir,
        "target": target_dir,
        "success": result.returncode == 0,
        "output": result.stdout,
        "error": result.stderr,
    }


def sync_directories(config, out_dir, method="parallel", max_workers=4, batch_size=5):
    """
    Sync directories using specified method.

    Args:
        config: Configuration dictionary
        out_dir: Output directory Path
        method: 'single' (one rsync command), 'separate' (sequential), or 'parallel'
        max_workers: Maximum number of parallel processes for 'parallel' method
        batch_size: Number of directory pairs per rsync command for 'single' method
    """
    for ablation_name, ablation_config in config["experiments"].items():
        console.print(f"Ablation: {ablation_name}")
        ablation_dir = out_dir / ablation_name
        ablation_dir.mkdir(parents=True, exist_ok=True)

        # Collect all valid source-target pairs
        sync_pairs = []
        for experiment in ablation_config["job_list"]:
            job_id = experiment["jobid"]
            source_dir = os.path.join(ablation_config["root_path"], str(job_id))

            if not os.path.exists(source_dir):
                console.print(f"[yellow]Job directory {source_dir} not found[/yellow]")
                continue

            target_dir = ablation_dir / str(job_id)
            target_dir.mkdir(parents=True, exist_ok=True)
            sync_pairs.append((source_dir, target_dir))

        if method == "single":
            # Batch process using single rsync commands
            for i in range(0, len(sync_pairs), batch_size):
                batch = sync_pairs[i : i + batch_size]
                rsync_args = []
                for source, target in batch:
                    rsync_args.extend([f"{source}/", f"{target}/"])

                cmd = [
                    "rsync",
                    "-av",
                    "--info=progress2",
                    "--ignore-existing",
                    "--exclude",
                    "*wandb*",
                ] + rsync_args

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    console.print(f"[green]Successfully synced batch {i//batch_size + 1}[/green]")
                else:
                    console.print(f"[red]Error in batch {i//batch_size + 1}[/red]")

        elif method == "parallel":
            # Parallel processing using multiple rsync processes
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(sync_single_directory, source, target)
                    for source, target in sync_pairs
                ]

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result["success"]:
                        console.print(f"[green]Successfully synced {result['source']}[/green]")
                    else:
                        console.print(
                            f"[red]Error syncing {result['source']}: {result['error']}[/red]"
                        )

        else:  # 'separate' method
            # Sequential processing
            for source, target in sync_pairs:
                result = sync_single_directory(source, target)
                if result["success"]:
                    console.print(f"[green]Successfully synced {result['source']}[/green]")
                else:
                    console.print(
                        f"[red]Error syncing {result['source']}: {result['error']}[/red]"
                    )


def main(config_file: str, out_dir: str):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found")
    with open(config_file) as f:
        config = yaml.safe_load(f)
        for ablation_name, ablation_config in config["experiments"].items():
            console.print(
                f"Ablation: {ablation_name}, Found {len(ablation_config['job_list'])} experiments"
            )

    console.print(f"Copying checkpoints to {out_dir}, experiment spec: {config_file}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sync_directories(config, out_dir, method="parallel", max_workers=4)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
