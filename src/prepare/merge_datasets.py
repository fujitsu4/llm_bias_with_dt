"""
merge_datasets.py
Author: Zakaria JOUILIL

Description:
    Merge list of datasets (2 or more)
Inputs:
    - Datasets to be merged (ex: MultiNLI, arXi etc.)

Outputs:
    - data/cleaned/merged_datasets.csv
    
Usage:
    !python -m src.prepare.merge_datasets \
    --inputs data/cleaned/snli_filtered.csv \
             data/cleaned/mnli_filtered.csv \
             data/cleaned/arxiv_filtered.csv \
    --output data/cleaned/merged_datasets.csv
"""

import argparse
import pandas as pd
from src.utils.paths import get_project_path

def main():
    parser = argparse.ArgumentParser(description="Merge two sentence CSV datasets with automatic reindexing.")
    parser.add_argument("--inputs", nargs="+", required=True,
                    help="List of CSV files to merge (2 or more)")
    parser.add_argument("--output", required=True, help="Path to merged output CSV")
    args = parser.parse_args()

    OUTPUT = get_project_path(*args.output.split("/"))
    INPUTS = [get_project_path(*p.split("/")) for p in args.inputs]

    dfs = [pd.read_csv(path, sep=";") for path in INPUTS]

    print("[INFO] Start merging datasets...]")
    
    # Reindex datasets
    current_index = 0
    for df in dfs:
        df["sentence_id"] = range(current_index, current_index + len(df))
        current_index += len(df)

    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(OUTPUT, sep=";", index=False)

    print("[INFO] Merged", " + ".join(str(len(df)) for df in dfs), "=", len(merged))
    print("[INFO] Saved to:", OUTPUT)

if __name__ == "__main__":
    main()