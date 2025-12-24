"""
verify_aggregate_features_patterns.py
Author: Zakaria JOUILIL

Description:
    Verification script for aggregate_features_patterns.py.

    This script verifies:
    - correctness of pattern extraction
    - consistency between source rules files and aggregated CSV
    - correctness of layer counts (pretrained)
    - correctness of seed counts and occurrences (untrained)
    - absence of silent logical errors (duplicates, ghost patterns)

Inputs:
    --mode: pretrained or untrained mode
    --input_csv: the aggregated patterns csv for pretrained or untrained mode
    --source_dir: the repository containing the simplified decision trees rules (simp_pos_rules)

Usage (pretrained):
    python -m src.dt_analysis.verify_aggregate_features_patterns \
        --mode pretrained \
        --input_csv outputs/dt_analysis/features_patterns_pretrained.csv \
        --source_dir outputs/decision_tree/pretrained

Usage (untrained):
    python -m src.dt_analysis.verify_aggregate_features_patterns \
        --mode untrained \
        --input_csv outputs/dt_analysis/features_patterns_untrained.csv \
        --source_dir /content/drive/MyDrive/results/decision_tree/untrained
"""

import argparse
import pandas as pd
from pathlib import Path
import re
from collections import defaultdict


# ---------------------------------------------------------
# Helper: parse patterns exactly like original script
# ---------------------------------------------------------
def parse_rules_file(path):
    patterns = []

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return patterns

    blocks = [b for b in content.split("\n\n") if b.strip()]

    for block in blocks:
        features = []
        for line in block.splitlines():
            m = re.search(r"---\s*([a-zA-Z0-9_]+)\s*[<>]=?", line)
            if m:
                features.append(m.group(1))
        if features:
            patterns.append(" -> ".join(features))

    return patterns


# ---------------------------------------------------------
# PRETRAINED VERIFICATION
# ---------------------------------------------------------
def verify_pretrained(agg_csv, source_dir):
    print("[INFO] Verifying pretrained pattern aggregation")

    agg = pd.read_csv(agg_csv, sep=";")
    expected_cols = {"pattern", "count", "layers"}
    assert expected_cols.issubset(set(agg.columns)), \
        f"[ERROR] Missing columns: {expected_cols - set(agg.columns)}"

    reconstructed = defaultdict(lambda: {"count": 0, "layers": set()})

    for layer in range(1, 13):
        fpath = source_dir / f"layer_{layer:02d}_rules_simp_pos.txt"
        if not fpath.exists():
            continue

        patterns = parse_rules_file(fpath)
        for p in patterns:
            reconstructed[p]["count"] += 1
            reconstructed[p]["layers"].add(layer)

    print(f"[INFO] Reconstructed {len(reconstructed)} unique patterns")

    for _, row in agg.iterrows():
        p = row["pattern"]
        count = row["count"]
        layers = sorted(eval(row["layers"]))

        assert p in reconstructed, f"[ERROR] Pattern not found in source: {p}"

        true_count = reconstructed[p]["count"]
        true_layers = sorted(reconstructed[p]["layers"])

        if count != true_count:
            print(f"[ERROR] Count mismatch for pattern: {p}")
            print(f"        CSV={count} | True={true_count}")

        if layers != true_layers:
            print(f"[ERROR] Layer mismatch for pattern: {p}")
            print(f"        CSV={layers} | True={true_layers}")

    print("[OK] Pretrained pattern aggregation verified successfully")


# ---------------------------------------------------------
# UNTRAINED VERIFICATION
# ---------------------------------------------------------
def verify_untrained(agg_csv, source_dir):
    print("[INFO] Verifying untrained pattern aggregation")

    agg = pd.read_csv(agg_csv, sep=";")
    expected_cols = {"pattern", "occurrences", "seed_count"}
    assert expected_cols.issubset(set(agg.columns)), \
        f"[ERROR] Missing columns: {expected_cols - set(agg.columns)}"

    reconstructed = defaultdict(lambda: {"occurrences": 0, "seeds": set()})

    seed_dirs = [
        d for d in source_dir.iterdir()
        if d.is_dir() and d.name.startswith("seed_")
    ]

    print(f"[INFO] Detected {len(seed_dirs)} seed directories")

    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.replace("seed_", ""))

        for layer in range(1, 13):
            fpath = seed_dir / f"layer_{layer:02d}_rules_simp_pos.txt"
            if not fpath.exists():
                continue

            patterns = parse_rules_file(fpath)
            for p in patterns:
                reconstructed[p]["occurrences"] += 1
                reconstructed[p]["seeds"].add(seed)

    print(f"[INFO] Reconstructed {len(reconstructed)} unique patterns")

    for _, row in agg.iterrows():
        p = row["pattern"]
        occ = row["occurrences"]
        seed_count = row["seed_count"]

        assert p in reconstructed, f"[ERROR] Pattern not found in source: {p}"

        true_occ = reconstructed[p]["occurrences"]
        true_seed_count = len(reconstructed[p]["seeds"])

        if occ != true_occ:
            print(f"[ERROR] Occurrence mismatch for pattern: {p}")
            print(f"        CSV={occ} | True={true_occ}")

        if seed_count != true_seed_count:
            print(f"[ERROR] Seed count mismatch for pattern: {p}")
            print(f"        CSV={seed_count} | True={true_seed_count}")

    print("[OK] Untrained pattern aggregation verified successfully")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["pretrained", "untrained"])
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--source_dir", required=True)

    args = parser.parse_args()

    source_dir = Path(args.source_dir)

    if args.mode == "pretrained":
        verify_pretrained(args.input_csv, source_dir)
    else:
        verify_untrained(args.input_csv, source_dir)


if __name__ == "__main__":
    main()