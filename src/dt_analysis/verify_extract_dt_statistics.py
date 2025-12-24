"""
verify_extract_dt_statistics.py
Author: Zakaria JOUILIL

Description:
    Verification script for extract_dt_statistics.py.
    This script performs consistency checks and logs potential issues

Inputs:
    --dt_analysis_dir: Directory containing dt_*.csv files (pretrained or untrained seeds)

Usage:
    python -m src.dt_analysis.verify_extract_dt_statistics \
        --dt_analysis_dir outputs/dt_analysis/dt_features_depth_pretrained.csv
"""

import pandas as pd
from pathlib import Path
import argparse


# ---------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------
def log_ok(msg):
    print(f"[OK] {msg}")

def log_warn(msg):
    print(f"[WARN] {msg}")

def log_err(msg):
    print(f"[ERROR] {msg}")


# ---------------------------------------------------------
# CHECK 1: pattern integrity
# ---------------------------------------------------------
def check_patterns(df):
    errors = 0

    for i, row in df.iterrows():
        pattern = row["pattern"]
        feats = pattern.split(" -> ")
        length = row["pattern_length"]

        if len(feats) != length:
            log_err(
                f"Pattern length mismatch at row {i}: "
                f"{len(feats)} != {length} | {pattern}"
            )
            errors += 1

        if feats[0].strip() == "":
            log_err(f"Empty root feature at row {i}")
            errors += 1

    if errors == 0:
        log_ok("All patterns have valid root and correct length")

    return errors


# ---------------------------------------------------------
# CHECK 2: depth consistency
# ---------------------------------------------------------
def check_depth_consistency(df_feat):
    errors = 0

    grouped = df_feat.groupby(["model", "seed", "layer"])

    for (model, seed, layer), g in grouped:
        depths = g["depth"].values
        if (depths < 1).any():
            log_err(f"Depth < 1 detected in {model}, seed={seed}, layer={layer}")
            errors += 1

    if errors == 0:
        log_ok("All depths are >= 1")

    return errors


# ---------------------------------------------------------
# CHECK 3: layer bounds
# ---------------------------------------------------------
def check_layers(df):
    if not df["layer"].between(1, 12).all():
        log_err("Layer index outside [1, 12]")
        return 1

    log_ok("All layers are within [1, 12]")
    return 0


# ---------------------------------------------------------
# CHECK 4: orphan features
# ---------------------------------------------------------
def check_orphan_features(df_feat, df_pat):
    errors = 0

    pat_features = set(
        f for p in df_pat["pattern"] for f in p.split(" -> ")
    )
    feat_features = set(df_feat["feature"].unique())

    orphan = feat_features - pat_features

    if orphan:
        log_warn(f"Orphan features detected (appear alone): {sorted(orphan)}")
    else:
        log_ok("No orphan features detected")

    return errors


# ---------------------------------------------------------
# MAIN VERIFICATION
# ---------------------------------------------------------
def main():
    print("=======================================")
    print(" VERIFYING extract_dt_statistics OUTPUT ")
    print("=======================================\n")

    csv_files = list(DT_ANALYSIS_DIR.glob("dt_*_*.csv"))

    if not csv_files:
        log_err("No CSV files found. Did you run extract_dt_statistics?")
        return

    for f in csv_files:
        print(f"\n[CHECK] {f.name}")

        if "patterns" in f.name:
            df = pd.read_csv(f, sep=";")
            if df.empty:
                log_warn("Patterns CSV is empty")
                continue

            e1 = check_patterns(df)
            e2 = check_layers(df)

        elif "features_depth" in f.name:
            df = pd.read_csv(f, sep=";")
            if df.empty:
                log_warn("Features CSV is empty")
                continue

            e3 = check_depth_consistency(df)

        else:
            log_warn("Unknown CSV type")

    print("\n=======================================")
    print(" VERIFICATION COMPLETED ")
    print("=======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dt_analysis_dir",
        required=True,
        help="Directory containing dt_*.csv files (pretrained or untrained seeds)"
    )
    args = parser.parse_args()

    DT_ANALYSIS_DIR = Path(args.dt_analysis_dir)

    main()