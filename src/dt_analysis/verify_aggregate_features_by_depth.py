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
    print("=" * 80)
    print("[INFO] Verifying PRETRAINED aggregation")
    print("[INFO] Aggregated CSV :", agg_csv)
    print("[INFO] Source CSV     :", source_csv)

    agg = pd.read_csv(agg_csv, sep=";")
    src = pd.read_csv(source_csv, sep=";")

    print(f"[INFO] Aggregated rows loaded : {len(agg)}")
    print(f"[INFO] Source rows loaded     : {len(src)}")

    expected_cols = {"Depth", "Feature", "Occurrence", "Layers"}
    missing = expected_cols - set(agg.columns)
    if missing:
        raise ValueError(f"[ERROR] Missing columns in aggregated CSV: {missing}")

    print("[INFO] Aggregated schema validated")

    print(f"[INFO] Unique depths   : {sorted(agg['Depth'].unique())}")
    print(f"[INFO] Unique features : {agg['Feature'].nunique()}")
    print(f"[INFO] Unique layers   : {sorted(src['layer'].unique())}")

    errors = 0

    for idx, row in agg.iterrows():
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
                f"[ERROR] Occurrence mismatch (depth={depth}, feature={feature}) : "
                f"reported={occ}, true={len(true_layers)}"
            )
            errors += 1

        if sorted(layers) != sorted(true_layers):
            print(
                f"[ERROR] Layer list mismatch (depth={depth}, feature={feature}) : "
                f"reported={layers}, true={sorted(true_layers)}"
            )
            errors += 1

        if idx == 0:
            print("[INFO] Example verification:")
            print(f"       Depth        : {depth}")
            print(f"       Feature      : {feature}")
            print(f"       Occurrence   : {occ}")
            print(f"       True layers  : {sorted(true_layers)}")

    print("-" * 80)
    print("[SUMMARY] PRETRAINED verification completed")
    print(f"          Depths verified   : {agg['Depth'].nunique()}")
    print(f"          Features verified : {agg['Feature'].nunique()}")
    print(f"          Total checks      : {len(agg)}")
    print(f"          Errors detected   : {errors}")

    if errors == 0:
        print("[OK] Pretrained aggregation verified successfully")
    else:
        print("[FAIL] Pretrained aggregation verification failed")


# ---------------------------------------------------------
# UNTRAINED VERIFICATION
# ---------------------------------------------------------
def verify_untrained(agg_csv, source_dir):
    print("=" * 80)
    print("[INFO] Verifying UNTRAINED aggregation")
    print("[INFO] Aggregated CSV :", agg_csv)
    print("[INFO] Source dir     :", source_dir)

    agg = pd.read_csv(agg_csv, sep=";")
    print(f"[INFO] Aggregated rows loaded : {len(agg)}")

    expected_cols = {"Depth", "Feature", "Seeds_Count"}
    missing = expected_cols - set(agg.columns)
    if missing:
        raise ValueError(f"[ERROR] Missing columns in aggregated CSV: {missing}")

    csv_files = glob.glob(
        os.path.join(source_dir, "dt_features_depth_untrained_seed_*.csv")
    )

    if not csv_files:
        raise ValueError("[ERROR] No source seed CSVs found")

    print(f"[INFO] Found {len(csv_files)} seed CSV files")

    all_dfs = []
    seeds = []

    for path in csv_files:
        df = pd.read_csv(path, sep=";")
        all_dfs.append(df)
        seeds.extend(df["seed"].unique().tolist())

    src = pd.concat(all_dfs, ignore_index=True)
    unique_seeds = sorted(set(seeds))

    print(f"[INFO] Unique seeds detected : {len(unique_seeds)}")
    print(f"[INFO] Seeds IDs (first 10)  : {unique_seeds[:10]}{' ...' if len(unique_seeds) > 10 else ''}")
    print(f"[INFO] Unique depths        : {sorted(agg['Depth'].unique())}")
    print(f"[INFO] Unique features      : {agg['Feature'].nunique()}")

    errors = 0

    for idx, row in agg.iterrows():
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
                f"[ERROR] Seed count mismatch (depth={depth}, feature={feature}) : "
                f"reported={seeds_count}, true={len(true_seeds)}"
            )
            errors += 1

        if idx == 0:
            print("[INFO] Example verification:")
            print(f"       Depth               : {depth}")
            print(f"       Feature             : {feature}")
            print(f"       Seeds_Count         : {seeds_count}")
            print(f"       True distinct seeds : {len(true_seeds)}")

    print("-" * 80)
    print("[SUMMARY] UNTRAINED verification completed")
    print(f"          Seeds verified    : {len(unique_seeds)}")
    print(f"          Depths verified   : {agg['Depth'].nunique()}")
    print(f"          Features verified : {agg['Feature'].nunique()}")
    print(f"          Total checks      : {len(agg)}")
    print(f"          Errors detected   : {errors}")

    if errors == 0:
        print("[OK] Untrained aggregation verified successfully")
    else:
        print("[FAIL] Untrained aggregation verification failed")


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