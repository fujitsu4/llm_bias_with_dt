"""
compute_attention_top5.py
Author: Zakaria JOUILIL

Description:
    Compute per-layer attention sums and top-5 flags for ALL tokens of ALL sentences.
    Uses ONLY compute_attention() (tested and validated before), and processes
    the entire dataset.

Inputs and paramters : 
    --model : choose pretrained or untrained configuration
    --seed : choose the fixed seed (for untrained model)
    --input_csv : the csv containg bert tokens (with features)
    --top_k : choose top_k token (default : top 5)
    --device : choose device (GPU)

Outputs:
    - outputs/attention/attention_top5_pretrained.csv
    - outputs/attention/attention_top5_untrained_seed_XXX.csv   (10 seeds)

Usage : 
    For pretained model :
        python -m src.attention.compute_attention_top5 \
        --model pretrained \
        --input_csv outputs/bert/bert_final_features.csv

    For untrained model (example with seed 10) : 
        python -m src.attention.compute_attention_top5 \
        --model untrained --seed 10 \
        --input_csv outputs/bert/bert_final_features.csv

"""

import argparse
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast

from src.attention.compute_attention_core import (
    load_model,
    compute_attention,
    fix_seed,
)


# ---------------------------------------------------------
# Utility: select top-k for 1D numpy array
# ---------------------------------------------------------
def topk_boolean_flags(arr, k=5):
    seq_len = arr.shape[0]
    kk = min(k, seq_len)
    idx_desc = np.argsort(-arr)
    topk_idx = idx_desc[:kk]
    flags = np.zeros(seq_len, dtype=int)
    flags[topk_idx] = 1
    return flags


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["pretrained", "untrained"], required=True)
    parser.add_argument("--seed", type=int, default=None, help="Seed required for untrained")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------------------------------
    # Model selection
    # -----------------------------------------------------
    if args.model == "pretrained":
        pretrained_flag = True
        print("[INFO] Loading PRETRAINED model ...")
    else:
        if args.seed is None:
            raise ValueError("Untrained model requires --seed")
        pretrained_flag = False
        print(f"[INFO] Loading RANDOMLY INITIALIZED model with seed {args.seed} ...")
        fix_seed(args.seed)

    model = load_model(pretrained_flag, seed=args.seed)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    model.to(device)
    model.eval()

    # -----------------------------------------------------
    # Load input CSV (with ALL 10k sentences, ALL tokens)
    # -----------------------------------------------------
    print(f"[INFO] Loading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv, sep=";", keep_default_na=False)

    # minimal columns required
    for col in ("sentence_id", "bert_index", "bert_token"):
        if col not in df.columns:
            raise ValueError(f"Missing column in input CSV: {col}")

    # -----------------------------------------------------
    # Prepare output directory
    # -----------------------------------------------------
    out_dir = "outputs/attention"
    os.makedirs(out_dir, exist_ok=True)

    if pretrained_flag:
        out_name = "attention_top5_pretrained.csv"
    else:
        out_name = f"attention_top5_untrained_seed_{args.seed:03d}.csv"

    out_path = os.path.join(out_dir, out_name)

    # -----------------------------------------------------
    # Add 24 output columns:
    # sum_l1 .. sum_l12  +  top5_l1 .. top5_l12
    # -----------------------------------------------------
    n_layers = 12  # BERT-base
    for L in range(1, n_layers + 1):
        df[f"sum_l{L}"] = np.nan
        df[f"top5_l{L}"] = 0

    # -----------------------------------------------------
    # Process ALL sentences
    # -----------------------------------------------------
    grouped = df.groupby("sentence_id", sort=False)
    print(f"[INFO] Processing {len(grouped)} sentences ...")

    for sid, group in tqdm(grouped, total=len(grouped), desc="Sentences"):
        # Sort tokens by bert_index
        group = group.sort_values("bert_index")
        token_list = group["bert_token"].astype(str).tolist()
        df_indices = group.index.tolist()

        # ------------------------
        # ATTENTION computation
        # ------------------------
        # feed EXACT tokens → no retokenization ever
        att = compute_attention(model, tokenizer, token_list)

        layer_sums = att["layer_sums"]   # list of 12 tensors (seq_len,)
        seq_len = len(token_list)

        # ------------------------
        # For each layer :
        #   1) save attention sum
        #   2) compute top-5 flags
        # ------------------------
        for li in range(n_layers):
            layer_sum_np = layer_sums[li].detach().cpu().numpy()  # (seq_len,)
            col_sum = f"sum_l{li+1}"
            col_top = f"top5_l{li+1}"

            # Set sums for each token
            for j, idx in enumerate(df_indices):
                df.at[idx, col_sum] = float(layer_sum_np[j])

            # Compute top-k flags within this sentence
            flags = topk_boolean_flags(layer_sum_np, k=args.top_k)
            for j, idx in enumerate(df_indices):
                df.at[idx, col_top] = int(flags[j])

    # -----------------------------------------------------
    # Enforce types
    # -----------------------------------------------------
    for L in range(1, n_layers + 1):
        df[f"sum_l{L}"] = df[f"sum_l{L}"].astype(float)
        df[f"top5_l{L}"] = df[f"top5_l{L}"].astype(int)

    # -----------------------------------------------------
    # Save output
    # -----------------------------------------------------
    print(f"[INFO] Saving final file: {out_path}")
    df.to_csv(out_path, sep=";", index=False)
    print("[OK] DONE — attention top5 computation complete.")


if __name__ == "__main__":
    main()