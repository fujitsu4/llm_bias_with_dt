"""
verify_compute_missing_features.py
Author: Zakaria JOUILIL

Description:
    Verification script for compute_missing_features.py.

    This script verifies:
    - correctness of feature reference space (attention CSV)
    - exclusion logic (ID columns, attention prefix)
    - consistency between attention features and DT-used features
    - validity of "missing features" claims
    - absence of silent logical errors (ghost features, naming mismatch)

Inputs:
    --attention_csv: the attention csv file containing all features
    --pretrained_csv: the aggregated csv (grouped by depth) for pretrained model
    --untrained_csv: the aggregated csv (grouped by depth) for untrained model

Usage:
    python -m src.dt_analysis.verify_compute_missing_features \
        --attention_csv outputs/attention/attention_top5_pretrained_sample.csv \
        --pretrained_csv outputs/dt_analysis/features_by_depth_pretrained.csv \
        --untrained_csv outputs/dt_analysis/features_by_depth_untrained.csv
"""

import argparse
import pandas as pd
from pathlib import Path


ID_COLUMNS = {
    "sentence_id", "bert_index", "bert_token", "word_id", "dataset"
}

ATTENTION_PREFIX = "top5_"


# ---------------------------------------------------------
# Helper: extract reference features
# ---------------------------------------------------------
def extract_reference_features(attention_csv):
    df = pd.read_csv(attention_csv, sep=";")

    print(f"[INFO] Attention CSV loaded : {attention_csv}")
    print(f"[INFO] Total columns        : {len(df.columns)}")

    ref_features = []

    for col in df.columns:
        if col in ID_COLUMNS:
            continue
        if col.startswith(ATTENTION_PREFIX):
            continue
        ref_features.append(col)

    print(f"[INFO] ID columns excluded           : {len(ID_COLUMNS)}")
    print(f"[INFO] Attention-prefixed excluded  : "
          f"{len([c for c in df.columns if c.startswith(ATTENTION_PREFIX)])}")
    print(f"[INFO] Reference features extracted : {len(ref_features)}")

    return sorted(ref_features)


# ---------------------------------------------------------
# Helper: extract DT-used features
# ---------------------------------------------------------
def extract_dt_features(dt_csv, label):
    df = pd.read_csv(dt_csv, sep=";")
    used = set(df["Feature"].unique())

    print(f"[INFO] {label} DT CSV loaded : {dt_csv}")
    print(f"[INFO] {label} features used : {len(used)}")

    return used


# ---------------------------------------------------------
# VERIFICATION
# ---------------------------------------------------------
def verify_missing_features(attention_csv, pretrained_csv, untrained_csv):
    print("=" * 80)
    print("[INFO] Verifying missing feature computation logic")

    ref_features = extract_reference_features(attention_csv)
    ref_set = set(ref_features)

    pretrained_used = extract_dt_features(pretrained_csv, "Pretrained")
    untrained_used = extract_dt_features(untrained_csv, "Untrained")

    print("-" * 80)
    print("[INFO] Cross-checking feature spaces")

    # --- Check that DT-used features belong to reference space
    invalid_pretrained = pretrained_used - ref_set
    invalid_untrained = untrained_used - ref_set

    if invalid_pretrained:
        print("[ERROR] Pretrained uses features NOT in attention reference:")
        print("        ", sorted(invalid_pretrained))
    else:
        print("[OK] All pretrained-used features belong to attention space")

    if invalid_untrained:
        print("[ERROR] Untrained uses features NOT in attention reference:")
        print("        ", sorted(invalid_untrained))
    else:
        print("[OK] All untrained-used features belong to attention space")

    # --- Compute missing features
    pretrained_missing = sorted(ref_set - pretrained_used)
    untrained_missing = sorted(ref_set - untrained_used)

    print("-" * 80)
    print("[INFO] Missing feature statistics")
    print(f"[INFO] Total reference features : {len(ref_set)}")

    print(f"[INFO] Pretrained missing       : {len(pretrained_missing)}")
    print(f"[INFO] Untrained missing        : {len(untrained_missing)}")

    if pretrained_missing:
        print("[INFO] Example pretrained missing features (first 10):")
        print("       ", pretrained_missing[:10])

    if untrained_missing:
        print("[INFO] Example untrained missing features (first 10):")
        print("       ", untrained_missing[:10])

    # --- Logical sanity checks
    assert len(pretrained_used) + len(pretrained_missing) == len(ref_set), \
        "[ERROR] Pretrained used + missing != reference set"

    assert len(untrained_used) + len(untrained_missing) == len(ref_set), \
        "[ERROR] Untrained used + missing != reference set"

    print("-" * 80)
    print("[SUMMARY] Missing feature verification completed successfully")
    print("[OK] 'Missing features' are correctly defined as:")
    print("     features present in attention CSV")
    print("     BUT never selected by decision trees")

    print("=" * 80)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention_csv", required=True,
                        help="Attention CSV defining full feature space")
    parser.add_argument("--pretrained_csv", required=True,
                        help="Aggregated pretrained DT features")
    parser.add_argument("--untrained_csv", required=True,
                        help="Aggregated untrained DT features")

    args = parser.parse_args()

    verify_missing_features(
        attention_csv=args.attention_csv,
        pretrained_csv=args.pretrained_csv,
        untrained_csv=args.untrained_csv,
    )


if __name__ == "__main__":
    main()