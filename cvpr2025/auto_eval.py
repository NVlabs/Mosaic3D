import csv
import os
import shutil
from typing import List

import yaml
from rich.console import Console
from rich.progress import track
from rich.table import Table

console = Console()


def run_evaluation(config: dict):
    for ablation_name, ablation_config in config.items():
        console.print(f"Ablation: {ablation_name}")

        for experiment in ablation_config["experiments"]:
            job_id = str(experiment["jobid"])
            csv_dir = os.path.join("results", job_id, "csv")
            if os.path.exists(csv_dir):
                console.print(f"Skipping {job_id} because csv dir already exists")
                continue
            slurm_log = os.path.join("slurm_outputs", f"{job_id}.log")
            training_args = None
            with open(slurm_log) as f:
                for line in f:
                    if line.strip().startswith("TRAINING_ARGS:"):
                        lines = line.strip()
                        training_args = lines.split("TRAINING_ARGS:")[1].strip()
                        break

            if training_args is None:
                raise ValueError(f"Training args not found in {slurm_log}")

            # replace logger
            training_args = training_args.replace(
                "logger=auto_resume_wandb",
                f"logger=csv logger.csv.save_dir=results/{job_id}",
            )
            training_args += " data.val_datasets.1.mask_dir=null"

            # find best checkpoint
            ckpt_path = os.path.join("results", job_id, "checkpoints")
            ckpts = os.listdir(ckpt_path)
            ckpts = list(filter(lambda x: x.startswith("epoch"), ckpts))
            if len(ckpts) == 0:
                raise ValueError(f"No checkpoint found in {ckpt_path}")
            best_ckpt = max(ckpts, key=lambda x: int(x.split(".ckpt")[0].split("_")[1]))
            best_ckpt = os.path.join(ckpt_path, best_ckpt)

            # run evaluation
            cmd = f"python src/eval.py {training_args} ckpt_path={best_ckpt}"
            console.print(cmd)

            # Redirect stdout to capture tqdm output
            os.environ["PYTHONUNBUFFERED"] = "1"  # Ensure Python output is unbuffered
            os.environ["TQDM_DISABLE"] = "0"  # Enable tqdm
            os.system(cmd)


def print_stat(
    config: dict,
    metrics: List[str] = [
        "epoch",
        # "val_scannet20/miou",
        # "val_scannet20/macc",
        "val_scannet200/miou",
        "val_scannet200/macc",
        # "val_scannet200/miou_all",
        # "val_scannet200/macc_all",
        "val_scannet200/miou_head",
        "val_scannet200/miou_common",
        "val_scannet200/miou_tail",
    ],
):
    for ablation_name, ablation_config in config.items():
        table = Table(title=f"Results for {ablation_name}")
        table.add_column("job_id")  # Add job_id column
        for metric in metrics:
            table.add_column(metric)

        # Initialize list to store all rows for combined DataFrame
        all_df_rows = []
        for experiment in track(ablation_config["experiments"]):
            job_id = str(experiment["jobid"])
            csv_dir = os.path.join("results", job_id, "csv")
            results = os.listdir(csv_dir)
            if len(results) == 0:
                raise ValueError(f"No results found in {csv_dir}")
            latest_result = max(results, key=lambda x: int(x.split("_")[-1]))
            csv_path = os.path.join(csv_dir, latest_result, "metrics.csv")

            with open(csv_path) as f:
                reader = csv.reader(f)
                header = next(reader)  # get header
                metric_indices = []
                for metric in metrics:
                    try:
                        metric_indices.append(header.index(metric))
                    except ValueError:
                        console.print(f"Warning: Metric {metric} not found in results")
                        metric_indices.append(-1)

                for row in reader:
                    table_row = [job_id]  # Add job_id as first column
                    for idx in metric_indices:
                        if idx == -1:
                            table_row.append("N/A")
                        else:
                            table_row.append(row[idx])
                    table.add_row(*table_row)
                    # Store row for combined DataFrame
                    all_df_rows.append(table_row)
        console.print(table)


def cleanup(config: dict):
    console.print("Removing csv logs")
    for ablation_name, ablation_config in config.items():
        for experiment in track(ablation_config["experiments"]):
            job_id = str(experiment["jobid"])
            csv_dir = os.path.join("results", job_id, "csv")
            if os.path.exists(csv_dir):
                shutil.rmtree(csv_dir)
                console.print(f"Removed {csv_dir}")


def main(
    config_file: str, run_eval: bool = False, run_print: bool = False, run_cleanup: bool = False
):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found")
    with open(config_file) as f:
        config = yaml.safe_load(f)
        for ablation_name, ablation_config in config.items():
            console.print(
                f"Ablation: {ablation_name}, Found {len(ablation_config['experiments'])} experiments"
            )

    if run_cleanup:
        cleanup(config)
    if run_eval:
        run_evaluation(config)
    if run_print:
        print_stat(config)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
