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
    --sentence_ids : (will ignore num_sentences) choose precise IDs of sentences to be tested

Outputs :
    Attention score matrixes (per sentence, per head)
    Sum of attention score matrixes (per sentence, per head and per layer).
    3 Sentences samples are stored at the "debug_samples" repository (fixed repository, not a parameter of the script)

Usage:
    For pretrained model : 
        python -m src.attention.debug_attention \
            --input_csv outputs/bert/bert_final_features.csv \
            --num_sentences 3 \
            --model pretrained

    For untrained:
        python -m src.attention.debug_attention \
            --input_csv outputs/bert/bert_final_features.csv \
            --num_sentences 3 \
            --model untrained --seed 123

    NOTE : If sentence_ids is choosen, usage for pretrained model will be :
        python -m src.attention.debug_attention \
            --input_csv outputs/bert/bert_final_features.csv \
            --model pretrained \
            --sentence_ids 2714 4131 4465 4562 4692 4773
"""

import argparse
import os
import pandas as pd
import torch
import numpy as np
from termcolor import colored

from src.attention.compute_attention_core import (
    load_model,
    compute_attention
)
from transformers import BertTokenizer

# -------------------------
# Test functions (internal)
# -------------------------
def test_attention_values_are_probabilities(data):
    att = data["attentions"]  # shape: (L, H, S, S)
    if not torch.all(att >= -1e-8):
        raise AssertionError("Attention contains negative values")
    if not torch.all(att <= 1.0 + 1e-6):
        raise AssertionError("Attention contains values > 1")

def test_attention_rows_sum_to_one(data, atol=1e-4):
    att = data["attentions"]  # (L, H, S, S)
    # For each layer/head, rows (source -> distribution over targets) must sum to 1
    row_sums = att.sum(dim=-1)  # shape (L, H, S)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=atol):
        # small debugging info
        max_dev = torch.max(torch.abs(row_sums - 1.0)).item()
        raise AssertionError(f"Some attention rows do not sum to 1 (max dev {max_dev:.6e})")

def test_matrix_matches_token_length(data, original_tokens):
    seq_len = len(original_tokens)
    att = data["attentions"]  # (L, H, S, S)
    if att.shape[2] != seq_len or att.shape[3] != seq_len:
        raise AssertionError(f"Matrix size mismatch: expected {seq_len}x{seq_len}, got {att.shape[2]}x{att.shape[3]}")

def test_tokens_preserved_order(data, original_tokens):
    returned_tokens = data["tokens"]
    if returned_tokens != original_tokens:
        # show difference for debug
        raise AssertionError(f"Tokens order changed!\noriginal: {original_tokens}\nreturned: {returned_tokens}")

def test_attention_shape(data, model):
    att = data["attentions"]
    L, H, S1, S2 = att.shape
    expected_L = model.config.num_hidden_layers
    expected_H = model.config.num_attention_heads
    if L != expected_L:
        raise AssertionError(f"Unexpected number of layers: {L} != {expected_L}")
    if H != expected_H:
        raise AssertionError(f"Unexpected number of heads: {H} != {expected_H}")
    if S1 != S2:
        raise AssertionError("Attention matrix is not square")


# -------------------------
# Save utilities
# -------------------------
def save_matrix_with_tokens(path, matrix, tokens):
    """
    matrix: 2D numpy array (seq_len, seq_len)
    tokens: list of tokens length seq_len
    """
    seq_len = len(tokens)
    with open(path, "w", encoding="utf8") as f:
        f.write("TOKEN\t" + "\t".join(tokens) + "\n")
        for i in range(seq_len):
            row = [f"{matrix[i, j]:.6f}" for j in range(seq_len)]
            f.write(tokens[i] + "\t" + "\t".join(row) + "\n")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--num_sentences", type=int, default = 3, required=False)
    parser.add_argument("--model", choices=["pretrained", "untrained"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sentence_ids", nargs="*", type=int, default=None)
    args = parser.parse_args()

    # Load CSV (tokens already BERT WordPiece)
    df = pd.read_csv(args.input_csv, sep=";", keep_default_na=False, na_values=[])
 
    df["bert_token"] = df["bert_token"].astype(str)

    groups = df.groupby("sentence_id", sort=False)
    # If specific sentence IDs were requested, filter them
    if args.sentence_ids:
        print(colored(f"[INFO] Using specific sentence_ids: {args.sentence_ids}", "cyan"))
        selected = []
        for sid in args.sentence_ids:
            if sid in groups.groups:
                selected.append((sid, groups.get_group(sid)))
            else:
                print(colored(f"[WARN] sentence_id {sid} not found in input CSV.", "yellow"))
        sentences = selected
    else:
        sentences = list(groups)[:args.num_sentences]

    print(colored(f"[INFO] Debugging {len(sentences)} sentences", "cyan"))

    # Load model + tokenizer
    model = load_model(args.model == "pretrained", seed=args.seed)
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    # Prepare debug folder (dynamic)
    if args.model == "pretrained":
        out_dir = "outputs/debug_samples/pretrained"
    else:
        out_dir = f"outputs/debug_samples/untrained/seed_{args.seed}"
    os.makedirs(out_dir, exist_ok=True)

    # Process each selected sentence
    for sent_idx, (sid, group) in enumerate(sentences):
        bert_tokens = group.sort_values("bert_index")["bert_token"].tolist()
        print(colored(f"\n[INFO] Sentence {sid} — {len(bert_tokens)} tokens", "yellow"))
        print(colored("TOKENS (original):", "green"), bert_tokens)

        # Detect OOV before calling compute_attention (for reporting)
        token_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        unk_id = tokenizer.unk_token_id
        oov_indices = [i for i, tid in enumerate(token_ids) if tid == unk_id]
        oov_tokens = [bert_tokens[i] for i in oov_indices]
        if len(oov_indices) > 0:
            print(colored(f"[WARN] Detected {len(oov_indices)} OOV tokens (mapped to [UNK]): {oov_tokens}", "red"))

        try:
            data = compute_attention(model, tokenizer, bert_tokens)
        except Exception as e:
            print(colored(f"[ERROR] compute_attention failed: {e}", "red"))
            continue

        if data.get("attentions", None) is None:
            print(colored("[ERROR] Model returned NULL attentions.", "red"))
            continue

        # Also compute the tokens actually used by the model (from ids)
        # replicate what core did (but safe): get ids then convert back
        model_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        model_tokens = tokenizer.convert_ids_to_tokens(model_ids)

        # Attach model_tokens to data for tests and saving
        data["model_tokens"] = model_tokens

        # Run internal tests (raise if any fail)
        try:
            test_attention_shape(data, model)
            test_matrix_matches_token_length(data, bert_tokens)
            test_tokens_preserved_order(data, bert_tokens)
            test_attention_values_are_probabilities(data)
            test_attention_rows_sum_to_one(data)
            print(colored("[OK] All internal attention tests passed for this sentence.", "green"))
        except AssertionError as e:
            print(colored(f"[FAILED TEST] Sentence {sid}: {e}", "red"))
            # continue to save debug outputs for inspection
        except Exception as e:
            print(colored(f"[ERROR] Unexpected exception during tests for sentence {sid}: {e}", "red"))

        # Save outputs (matrices + sums + report)
        sentence_dir = os.path.join(out_dir, f"sentence_{sid}")
        os.makedirs(sentence_dir, exist_ok=True)

        attentions = data["attentions"]  # tensor (L, H, S, S)
        num_layers, num_heads, seq_len, _ = attentions.shape

        # Write per-layer-per-head matrices (with BOTH original tokens and model tokens)
        for L in range(num_layers):
            for H in range(num_heads):
                arr = attentions[L][H].detach().cpu().numpy()
                # file with original tokens as header/rows
                path_orig = os.path.join(sentence_dir, f"layer_{L+1:02d}_head_{H+1:02d}_orig.txt")
                save_matrix_with_tokens(path_orig, arr, bert_tokens)
                # file with model tokens as header/rows
                path_model = os.path.join(sentence_dir, f"layer_{L+1:02d}_head_{H+1:02d}_modeltokens.txt")
                save_matrix_with_tokens(path_model, arr, model_tokens)

        # Save layer sums (one file)
        for L, layer_sum in enumerate(data["layer_sums"]):
            arr = layer_sum.detach().cpu().numpy()
            path = os.path.join(sentence_dir, f"layer_{L+1:02d}_SUM.txt")
            with open(path, "w", encoding="utf8") as f:
                f.write(" ".join(f"{x:.6f}" for x in arr.tolist()))

        # Save head sums
        for L in range(num_layers):
            for H in range(num_heads):
                head_sum = data["head_sums"][L][H].detach().cpu().numpy()
                path = os.path.join(sentence_dir, f"layer_{L+1:02d}_head_{H+1:02d}_SUM.txt")
                with open(path, "w", encoding="utf8") as f:
                    f.write(" ".join(f"{x:.6f}" for x in head_sum.tolist()))

        # Save a small report (tokens, OOV, test status)
        report_path = os.path.join(sentence_dir, "report.txt")
        with open(report_path, "w", encoding="utf8") as f:
            f.write(f"sentence_id: {sid}\n")
            f.write(f"num_tokens: {len(bert_tokens)}\n")
            f.write("original_tokens: " + " ".join(bert_tokens) + "\n")
            f.write("model_tokens: " + " ".join(model_tokens) + "\n")
            if len(oov_tokens) > 0:
                f.write("OOV_tokens: " + " ".join(oov_tokens) + "\n")
            f.write("\nNote: All attention matrices are stored with both original and model-token headers.\n")

        print(colored(f"→ Debug results saved in {sentence_dir}", "cyan"))


if __name__ == "__main__":
    main()