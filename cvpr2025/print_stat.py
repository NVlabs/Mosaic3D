import glob
import itertools

import numpy as np
import pandas as pd


def main(exp_name: str, dataset: str, category_wise: bool = False):
    result_dir = f"./eval_rebuttal/{exp_name}"

    csv_files = glob.glob(f"{result_dir}/**/**/*.csv")

    assert len(csv_files) == 1, f"Found multiple csv files in {result_dir} {len(csv_files)}"

    csv_file = csv_files[0]
    df = pd.read_csv(csv_file)

    columns = []
    for metric in ["miou", "macc"]:
        columns.append(f"val_{dataset}/{metric}")

    if category_wise:
        for metric, category in itertools.product(["miou", "macc"], ["head", "common", "tail"]):
            columns.append(f"val_{dataset}/{metric}_{category}")

    print(df[columns])


if __name__ == "__main__":
    import fire

    fire.Fire(main)
