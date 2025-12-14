CHNGER LES DEPTH !!!!!! (2 à 5)
"""
extract_dt_statistics.py
Author: Zakaria JOUILIL

Description:
    Parse simplified positive decision tree rules (rules_simp_pos)
    and extract, PER MODEL / PER SEED:
        - discriminative features per depth and per layer
        - full decision patterns (feature paths)

This script:
    - works for pretrained BERT (single directory)
    - works for untrained BERT (multiple seed_xxxxxx subdirectories)
    - generates ONE CSV PER SEED (Option A)

It intentionally does NOT rely on rules_full and therefore does NOT
extract support or pos_rate.

Inputs and parameters :
    --input_dir : directory containing 12 decision trees (one per layer)
    --output_dir : directory where CSVs will be saved

Ouputs (for pretrained model OR for a seed of untrained model):
    dt_features_depth.csv : displays the discriminative features (per layer and per depth)
    dt_patterns.csv : displays patterns of discriminative features (per layer)

Usage :
    python -m src.dt_analysis.extract_dt_statistics \
    --input_dir outputs/decision_tree/pretrained
"""

import argparse
import re
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Parse one simplified rules file
# ---------------------------------------------------------
def parse_rules_file(path, layer, model, seed=None):
    rows_features = []
    rows_patterns = []

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return rows_features, rows_patterns

    blocks = [b for b in content.split("\n\n") if b.strip()]

    for block in blocks:
        lines = block.splitlines()
        features_in_path = []

        for line in lines:
            depth = line.count("|")

            m = re.search(r"---\s*([a-zA-Z0-9_]+)\s*[<>]=?", line)
            if m:
                feat = m.group(1)
                features_in_path.append(feat)

                rows_features.append({
                    "model": model,
                    "seed": seed,
                    "layer": layer,
                    "depth": depth,
                    "feature": feat,
                })

        if features_in_path:
            rows_patterns.append({
                "model": model,
                "seed": seed,
                "layer": layer,
                "pattern": " -> ".join(features_in_path),
                "pattern_length": len(features_in_path),
            })

    return rows_features, rows_patterns


# ---------------------------------------------------------
# Process ONE directory (pretrained OR one seed)
# ---------------------------------------------------------
def process_one_model(input_dir, model, seed, output_dir):
    all_features = []
    all_patterns = []

    for layer in range(1, 13):
        fname = f"layer_{layer:02d}_rules_simp_pos.txt"
        fpath = input_dir / fname

        if not fpath.exists():
            print(f"[WARN] Missing file: {fpath}")
            continue

        feats, pats = parse_rules_file(
            fpath, layer=layer, model=model, seed=seed
        )

        all_features.extend(feats)
        all_patterns.extend(pats)

    if not all_features:
        print(f"[WARN] No rules found in {input_dir}")
        return

    # ------------------------------
    # FEATURES
    # ------------------------------
    df_features = pd.DataFrame(all_features)
    df_features = (
        df_features.drop_duplicates()
        .sort_values(by=["layer", "depth", "feature"])
        .reset_index(drop=True)
    )

    # ------------------------------
    # PATTERNS
    # ------------------------------
    df_patterns = pd.DataFrame(all_patterns)

    suffix = "pretrained" if seed is None else f"untrained_seed_{seed:06d}"

    f_feat = output_dir / f"dt_features_depth_{suffix}.csv"
    f_pat = output_dir / f"dt_patterns_{suffix}.csv"

    df_features.to_csv(f_feat, index=False)
    df_patterns.to_csv(f_pat, index=False)

    print(f"[OK] Saved {f_feat}")
    print(f"[OK] Saved {f_pat}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="outputs/decision_tree/pretrained OR untrained")
    parser.add_argument("--output_dir", default="outputs/dt_analysis",
                        help="Directory where CSVs will be saved")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # PRETRAINED
    # -----------------------------------------------------
    if any(p.name.startswith("layer_") for p in input_dir.iterdir()):
        print("[INFO] Detected PRETRAINED model")
        process_one_model(
            input_dir=input_dir,
            model="pretrained",
            seed=None,
            output_dir=output_dir
        )
        return

    # -----------------------------------------------------
    # UNTRAINED (seed directories)
    # -----------------------------------------------------
    print("[INFO] Detected UNTRAINED model")

    for d in sorted(input_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("seed_"):
            continue

        seed = int(d.name.replace("seed_", ""))

        out_feat = output_dir / f"dt_features_depth_untrained_seed_{seed:06d}.csv"
        out_pat = output_dir / f"dt_patterns_untrained_seed_{seed:06d}.csv"

        if out_feat.exists() and out_pat.exists():
            print(f"[SKIP] Seed {seed} already processed")
            continue

        print(f"[RUN] Processing seed {seed}")
        process_one_model(
            input_dir=d,
            model="untrained",
            seed=seed,
            output_dir=output_dir
        )


if __name__ == "__main__":
    main()