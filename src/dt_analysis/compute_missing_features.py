"""
compute_missing_features.py
Author: Zakaria JOUILIL

Description:
    Compare features analyzed vs features never selected
    in decision tree results for pretrained and untrained BERT.

Inputs :
    --attention_csv : the attention csv file containing all features
    --pretrained_csv : aggregated features by depth file (pretrained model)
    --untrained_csv : aggregated features by depth file (untrained model)

Output :
    --log_file : log file containing missing features for both pretrained and untrained model

NOTE:
    The attention file is required as an argument to retrieve the exact names of all features.
    The attention file is preferred over the bert final features file because the latter contains
    duplicate features (both string and ID) used for debug.

Usage:
    python -m src.dt_analysis.compute_missing_features \
        --attention_csv outputs/attention/attention_top5_pretrained_sample.csv \
        --pretrained_csv outputs/dt_analysis/features_by_depth_pretrained.csv \
        --untrained_csv outputs/dt_analysis/features_by_depth_untrained.csv \
        --log_file logs/missing_features_comparison.txt
"""

import argparse
import pandas as pd
from datetime import datetime


ID_COLUMNS = {
    "sentence_id", "bert_index", "bert_token", "word_id", "dataset"
}

ATTENTION_PREFIX = "top5_"


def extract_all_features(attention_csv):
    df = pd.read_csv(attention_csv, sep = ";")
    features = []

    for col in df.columns:
        if col in ID_COLUMNS:
            continue
        if col.startswith(ATTENTION_PREFIX):
            continue
        features.append(col)

    return sorted(features)


def extract_used_features(dt_csv):
    df = pd.read_csv(dt_csv, sep = ";")
    return set(df["Feature"].unique())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention_csv", required=True)
    parser.add_argument("--pretrained_csv", required=True)
    parser.add_argument("--untrained_csv", required=True)
    parser.add_argument("--log_file", required=True)
    args = parser.parse_args()

    all_features = extract_all_features(args.attention_csv)
    pretrained_used = extract_used_features(args.pretrained_csv)
    untrained_used = extract_used_features(args.untrained_csv)

    pretrained_missing = sorted(set(all_features) - pretrained_used)
    untrained_missing = sorted(set(all_features) - untrained_used)

    with open(args.log_file, "w", encoding="utf8") as f:
        f.write(f"=== Feature Comparison Log ===\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

        f.write(f"Total features analyzed: {len(all_features)}\n")
        f.write(", ".join(all_features) + "\n\n")

        f.write("=== Pretrained BERT: Features NEVER selected ===\n")
        f.write(f"Total Missing Features: {len(pretrained_missing)}\n")
        f.write(", ".join(pretrained_missing) + "\n\n")

        f.write("=== Untrained BERT: Features NEVER selected ===\n")
        f.write(f"Total Missing Features: {len(untrained_missing)}\n")
        f.write(", ".join(untrained_missing) + "\n")

    print(f"[OK] Comparison log written to {args.log_file}")


if __name__ == "__main__":
    main()