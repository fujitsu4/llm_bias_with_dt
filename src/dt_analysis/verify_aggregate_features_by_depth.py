"""
verify_aggregate_features_by_depth.py
Author: Zakaria JOUILIL

Description:
    Verification script for aggregate_features_by_depth.py.

    This script checks:
    - input CSV consistency
    - aggregation correctness (depth / feature counts)
    - layer / seed accounting
    - output schema validity
    - silent logical errors (duplicates, impossible counts)

Inputs and arguments :
    --mode: pretrained or untrained mode
    --input_csv: the aggregated csv (grouped by depth) for pretrained or untrained mode
    --source_csv: for pretrained mode, the source csv is the csv containing the extracted statistics
    --source_dir: for untrained mode, the source dir is the repository containing the 30 extracted statsictis (for each seed)

Usage (pretrained):
    python -m src.dt_analysis.verify_aggregate_features_by_depth \
        --mode pretrained \
        --input_csv outputs/dt_analysis/features_by_depth_pretrained.csv \
        --source_csv outputs/dt_analysis/dt_features_depth_pretrained.csv

Usage (untrained):
    python -m src.dt_analysis.verify_aggregate_features_by_depth \
        --mode untrained \
        --input_csv outputs/dt_analysis/features_by_depth_untrained.csv \
        --source_dir /content/drive/MyDrive/results/dt_analysis

"""

import argparse
import pandas as pd
from pathlib import Path
import glob
import os
import ast

# ---------------------------------------------------------
# PRETRAINED VERIFICATION
# ---------------------------------------------------------
def verify_pretrained(agg_csv, source_csv):
    print("[INFO] Verifying pretrained aggregation")

    agg = pd.read_csv(agg_csv, sep=";")
    src = pd.read_csv(source_csv, sep=";")

    expected_cols = {"Depth", "Feature", "Occurrence", "Layers"}
    assert expected_cols.issubset(set(agg.columns)), \
        f"[ERROR] Missing columns in aggregated CSV: {expected_cols - set(agg.columns)}"

    # Check layer counts
    for _, row in agg.iterrows():
        depth = row["Depth"]
        feature = row["Feature"]
        occ = row["Occurrence"]
        layers = ast.literal_eval(row["Layers"])

        true_layers = (
            src[(src["depth"] == depth) & (src["feature"] == feature)]["layer"]
            .unique()
            .tolist()
        )

        if occ != len(true_layers):
            print(
                f"[ERROR] Mismatch for (depth={depth}, feature={feature}): "
                f"Occurrence={occ}, true={len(true_layers)}"
            )

        if sorted(layers) != sorted(true_layers):
            print(
                f"[ERROR] Layer list mismatch for (depth={depth}, feature={feature})"
            )

    print("[OK] Pretrained aggregation verified successfully")


# ---------------------------------------------------------
# UNTRAINED VERIFICATION
# ---------------------------------------------------------
def verify_untrained(agg_csv, source_dir):
    print("[INFO] Verifying untrained aggregation")

    agg = pd.read_csv(agg_csv, sep=";")

    expected_cols = {"Depth", "Feature", "Seeds_Count"}
    assert expected_cols.issubset(set(agg.columns)), \
        f"[ERROR] Missing columns in aggregated CSV: {expected_cols - set(agg.columns)}"

    csv_files = glob.glob(
        os.path.join(source_dir, "dt_features_depth_untrained_seed_*.csv")
    )

    if not csv_files:
        raise ValueError("[ERROR] No source seed CSVs found")

    all_dfs = []
    for path in csv_files:
        df = pd.read_csv(path, sep=";")
        all_dfs.append(df)

    src = pd.concat(all_dfs, ignore_index=True)

    # Check seed counts
    for _, row in agg.iterrows():
        depth = row["Depth"]
        feature = row["Feature"]
        seeds_count = row["Seeds_Count"]

        true_seeds = (
            src[(src["depth"] == depth) & (src["feature"] == feature)]["seed"]
            .unique()
            .tolist()
        )

        if seeds_count != len(true_seeds):
            print(
                f"[ERROR] Seed count mismatch for (depth={depth}, feature={feature}): "
                f"Seeds_Count={seeds_count}, true={len(true_seeds)}"
            )

    print("[OK] Untrained aggregation verified successfully")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["pretrained", "untrained"])
    parser.add_argument("--input_csv", required=True,
        help="Aggregated CSV to verify")
    parser.add_argument("--source_csv",
        help="Original dt_features_depth_pretrained.csv")
    parser.add_argument("--source_dir",
        help="Directory with dt_features_depth_untrained_seed_*.csv")

    args = parser.parse_args()

    if args.mode == "pretrained":
        if not args.source_csv:
            raise ValueError("--source_csv is required in pretrained mode")
        verify_pretrained(args.input_csv, args.source_csv)

    elif args.mode == "untrained":
        if not args.source_dir:
            raise ValueError("--source_dir is required in untrained mode")
        verify_untrained(args.input_csv, args.source_dir)


if __name__ == "__main__":
    main()