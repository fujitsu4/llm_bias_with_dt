"""
extract_dt_statistics.py
Author: Zakaria JOUILIL

Description:
    Parse simplified positive decision tree rules (rules_simp_pos)
    and extract:
        - feature usage per depth and per layer
        - full decision patterns (paths)

This script works for:
    - pretrained BERT (single directory)
    - untrained BERT (multiple seed_xxxxxx subdirectories)

It does NOT rely on rules_full and therefore does NOT extract
support or pos_rate by design.

Inputs and parameters : 
    --input_dir : directory containing 12 decision trees (one per layer)
    --output_dir : directory where CSVs will be saved

Ouputs (for pretrained model OR for a seed of untrained model):
    dt_features_depth.csv : displays the discriminative features (per layer and per depth)
    dt_patterns.csv : displays patterns of discriminative features (per layer) 

Usage :
    python -m src.analysis.extract_dt_statistics \
    --input_dir outputs/decision_tree/pretrained 
"""

import argparse
import os
import re
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Parse one simplified rules file
# ---------------------------------------------------------
def parse_rules_file(path, layer, model, seed=None):
    """
    Extract:
        - features with their depth
        - full patterns (feature1 -> feature2 -> ...)
    """
    rows_features = []
    rows_patterns = []

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return rows_features, rows_patterns

    blocks = [b for b in content.split("\n\n") if b.strip()]

    for block in blocks:
        lines = block.splitlines()
        features = []

        for line in lines:
            # depth = number of "|"
            depth = line.count("|")
            m = re.search(r"---\s*([a-zA-Z0-9_]+)\s*[<>]=?", line)
            if m:
                feat = m.group(1)
                features.append(feat)

                rows_features.append({
                    "model": model,
                    "seed": seed,
                    "layer": layer,
                    "depth": depth,
                    "feature": feat,
                })

        if features:
            rows_patterns.append({
                "model": model,
                "seed": seed,
                "layer": layer,
                "pattern": " -> ".join(features),
                "pattern_length": len(features),
            })

    return rows_features, rows_patterns


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Root directory (pretrained or untrained)")
    parser.add_argument("--output_dir", default="outputs/dt_analysis",
                        help="Directory where CSVs will be saved")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_features = []
    all_patterns = []

    # -----------------------------------------------------
    # Detect pretrained vs untrained
    # -----------------------------------------------------
    if any(p.name.startswith("layer_") for p in input_dir.iterdir()):
        # pretrained
        print("[INFO] Detected PRETRAINED directory")
        dirs = [(input_dir, "pretrained", None)]
    else:
        # untrained
        print("[INFO] Detected UNTRAINED directory")
        dirs = []
        for d in input_dir.iterdir():
            if d.is_dir() and d.name.startswith("seed_"):
                seed = int(d.name.replace("seed_", ""))
                dirs.append((d, "untrained", seed))

    # -----------------------------------------------------
    # Parse all layers
    # -----------------------------------------------------
    for d, model, seed in dirs:
        print(f"[INFO] Processing {model} {'' if seed is None else f'seed {seed}'}")

        for layer in range(1, 13):
            fname = f"layer_{layer:02d}_rules_simp_pos.txt"
            fpath = d / fname

            if not fpath.exists():
                print(f"[WARN] Missing file: {fpath}")
                continue

            feats, pats = parse_rules_file(
                fpath, layer=layer, model=model, seed=seed
            )

            all_features.extend(feats)
            all_patterns.extend(pats)

    # -----------------------------------------------------
    # Save CSVs
    # -----------------------------------------------------
    df_features = pd.DataFrame(all_features)

    # keep only what is needed for qualitative analysis
    df_features = (
    df_features.drop_duplicates()
    .sort_values(by=["layer", "depth"])
    .reset_index(drop=True)
    )

    df_patterns = pd.DataFrame(all_patterns)

    if model == "pretrained":
        suffix = "pretrained"
    else:
        suffix = f"untrained_seed_{seed}"
    
    f1 = output_dir / f"dt_features_depth_{suffix}.csv"
    f2 = output_dir / f"dt_patterns_{suffix}.csv"

    df_features.to_csv(f1, index=False)
    df_patterns.to_csv(f2, index=False)

    print(f"[OK] Saved: {f1}")
    print(f"[OK] Saved: {f2}")


if __name__ == "__main__":
    main()