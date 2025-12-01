"""
debug_attention.py
Author : Zakaria JOUILIL

Description:
    Debug tool to inspect attention matrices produced by compute_attention_core.
    This script generates human-readable logs for verification and validation.

Inputs and parameters :
    --input_csv : the csv containg bert tokens (with features)
    --num_sentences : number of sentences to be tested
    --model : choose pretrained or untrained configuration
    --seed : choose the fixed seed (for untrained model)

Outputs :
    The results are stored at the "debug_examples" repository (fixed repository, not a parameter of the script)

Usage:
    For pretrained model : 
        python debug_attention.py \
            --input_csv outputs/bert/bert_final_features.csv \
            --num_sentences 3 \
            --model pretrained

    For untrained:
        python debug_attention.py \
            --input_csv outputs/bert/bert_final_features.csv \
            --num_sentences 3 \
            --model untrained --seed 123
"""

import argparse
import os
import pandas as pd
from termcolor import colored

from compute_attention_core import (
    load_model,
    compute_attention
)
from transformers import BertTokenizer


def save_matrix(path, matrix):
    with open(path, "w", encoding="utf8") as f:
        for row in matrix:
            f.write(" ".join(f"{x:.6f}" for x in row.tolist()) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--num_sentences", type=int, required=True)
    parser.add_argument("--model", choices=["pretrained", "untrained"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.input_csv, sep=";")
    df["bert_token"] = df["bert_token"].astype(str)

    groups = df.groupby("sentence_id", sort=False)
    sentences = list(groups)[:args.num_sentences]

    print(colored(f"[INFO] Debugging {len(sentences)} sentences", "cyan"))

    # Load model + tokenizer
    model = load_model(args.model == "pretrained", seed=args.seed)
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    # Prepare debug folder
    out_dir = "outputs/debug_examples"
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Process each selected sentence
    # ---------------------------------------------------------
    for sent_idx, (sid, group) in enumerate(sentences):
        bert_tokens = group["bert_token"].tolist()

        print(colored(f"\n[INFO] Sentence {sid} — {len(bert_tokens)} tokens", "yellow"))
        print(colored("TOKENS:", "green"), bert_tokens)

        data = compute_attention(model, tokenizer, bert_tokens)

        # Save matrices layer by layer
        attentions = data["attentions"]
        num_layers, num_heads, seq_len, _ = attentions.shape

        sentence_dir = os.path.join(out_dir, f"sentence_{sid}")
        os.makedirs(sentence_dir, exist_ok=True)

        # ---------------------------------------------
        # Write each LAYER × HEAD matrix
        # ---------------------------------------------
        for L in range(num_layers):
            for H in range(num_heads):
                path = os.path.join(
                    sentence_dir,
                    f"layer_{L+1:02d}_head_{H+1:02d}.txt"
                )
                save_matrix(path, attentions[L][H])
        
        # ---------------------------------------------
        # Write layer sums
        # ---------------------------------------------
        for L, layer_sum in enumerate(data["layer_sums"]):
            path = os.path.join(sentence_dir, f"layer_{L+1:02d}_SUM.txt")
            with open(path, "w") as f:
                f.write(" ".join(f"{x:.6f}" for x in layer_sum.tolist()))

        # ---------------------------------------------
        # Write head sums
        # ---------------------------------------------
        for L in range(num_layers):
            for H in range(num_heads):
                head_sum = data["head_sums"][L][H]
                path = os.path.join(
                    sentence_dir,
                    f"layer_{L+1:02d}_head_{H+1:02d}_SUM.txt"
                )
                with open(path, "w") as f:
                    f.write(" ".join(f"{x:.6f}" for x in head_sum.tolist()))

        print(colored(
            f"→ Debug results saved in {sentence_dir}",
            "cyan"
        ))


if __name__ == "__main__":
    main()