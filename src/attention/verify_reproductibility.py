"""
verify_reproductibility.py
Author: Zakaria JOUILIL

Description :
    This script checks reproducibility of attention extraction using:
       - Pretrained BERT
       - Untrained BERT :
            With fixed seed (determinism)
            With multiple seeds (sensitivity test)

It relies ONLY on compute_attention_core.py


Inputs and parameters :
    --input_csv : the csv containg bert tokens (with features)
    --model : choose pretrained or untrained configuration
    --seed : choose the fixed seed (for untrained model)
    --layer : layer to be tested (or all layers)
    --iterations : number of test iterations
    --num_sentences : number of sentences to be tested
    --sentence_ids : (will ignore num_sentences) choose precise IDs of sentences to be tested
    --tolerance : tolerance threshold (default=1e-6)
    --compare_seeds : parameter to sensitivity test for untrained bert (different seeds gives different results)
    
Usage :
    python -m src.attention.verify_reproducibility \
    --input_csv outputs/bert/bert_final_features.csv \
    --model pretrained|untrained [--seed SEED] \
    --layer 4|all \
    --iterations 5 \
    [--num_sentences 3] \
    [--sentence_ids 2714 4131 4465] \
    --tolerance 1e-6 \
    [--compare_seeds]  # Sensitivity test for untrained Bert

"""

import argparse
import numpy as np
import torch
from src.attention.compute_attention_core import (
    load_model,
    compute_attention
)
from transformers import BertTokenizerFast
import pandas as pd

# -------------------------------------------------------------------
# Load CSV and extract sentences by:
#   - --sentence_ids or
#   - --num_sentences first N rows
# -------------------------------------------------------------------
""""
def load_sentences(csv_path, sentence_ids, num_sentences):
    df = pd.read_csv(csv_path, sep=";", keep_default_na=False, na_values=[])

    if sentence_ids:
        df = df[df["sentence_id"].isin(sentence_ids)]
        df = df.sort_values(by=["sentence_id", "bert_index"])
    else:
        df = df.head(num_sentences)

    if df.empty:
        raise ValueError("No sentences selected for reproducibility test.")

    return df
"""

def load_bert_tokens_from_csv(csv_path):

    df = pd.read_csv(csv_path, sep=";", keep_default_na=False, na_values=[])

    if "sentence_id" not in df.columns or "bert_token" not in df.columns:
        raise ValueError("CSV must contain columns 'sentence_id' and 'bert_token'")

    df = df.sort_values(by=["sentence_id", "bert_index"])

    grouped = {}
    for sid, group in df.groupby("sentence_id"):
        grouped[sid] = group["bert_token"].tolist()

    return grouped

# -------------------------------------------------------------------
# Compute layer summaries for 1 run
# Returns dict: {layer -> array(seq_len), ...}
# -------------------------------------------------------------------
def extract_layer_sums(model, tokenizer, tokens, layer):
    data = compute_attention(model, tokenizer, tokens)
    attentions = data["attentions"]  # (L, H, S, S)
    layer_sums = data["layer_sums"]

    if layer != "all":
        layer_idx = int(layer) - 1  # converts layer "4" → index 3
        return {layer_idx: layer_sums[layer_idx]}
    else:
        return {L: layer_sums[L] for L in range(len(layer_sums))}


# -------------------------------------------------------------------
# Compare two runs (float closeness + discrete top5)
# -------------------------------------------------------------------
def compare_runs(runA, runB, tolerance):
    results = {}

    for layer in runA:
        vA = np.array(runA[layer])
        vB = np.array(runB[layer])

        # float closeness
        float_ok = np.allclose(vA, vB, atol=tolerance)

        # discrete top5
        topA = np.argsort(vA)[-5:]
        topB = np.argsort(vB)[-5:]
        top_ok = np.array_equal(topA, topB)

        # max abs diff (for reporting)
        maxdiff = float(np.max(np.abs(vA - vB)))

        results[layer] = {
            "float_ok": float_ok,
            "top5_ok": top_ok,
            "maxdiff": maxdiff,
            "topA": topA,
            "topB": topB
        }

    return results


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", required=True)
    
    parser.add_argument("--model", choices=["pretrained", "untrained"], required=True)
    parser.add_argument("--seed", type=int)

    parser.add_argument("--layer", default="all")
    parser.add_argument("--iterations", type=int, default=3)

    parser.add_argument("--num_sentences", type=int, default=3)
    parser.add_argument("--sentence_ids", nargs="*", type=int)

    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--compare_seeds", action="store_true",
                        help="Test sensitivity: compare multiple random seeds for untrained model.")

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------------
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    # --------------------------------------------------------------
    # Prepare sentence set
    # --------------------------------------------------------------
    # Load CSV and reconstruct BERT tokens for all sentences
    bert_data = load_bert_tokens_from_csv(args.input_csv)

    # Determine which sentence IDs to test (either explicit list or first N available)
    if args.sentence_ids:
        # user provided explicit ids -> validate they exist
        requested_ids = args.sentence_ids
        missing = [sid for sid in requested_ids if sid not in bert_data]
        if missing:
            raise ValueError(f"Sentence IDs {missing} not found in CSV.")
        sentence_ids = requested_ids
    else:
        # take the first `num_sentences` available sentence_ids from the CSV
        available_ids = list(bert_data.keys())
        if len(available_ids) == 0:
            raise ValueError("No sentence_ids found in CSV.")
        sentence_ids = available_ids[: args.num_sentences]

    # Build the list of token lists (one entry per sentence_id), preserving order
    sentences = [bert_data[sid] for sid in sentence_ids]
    print(f"[DEBUG] Using sentence_ids: {sentence_ids}")
    for sid, s in zip(sentence_ids, sentences):
        print(f"[DEBUG] id={sid}, first tokens: {s[:10]}")

    print("\n============================================")
    print("REPRODUCIBILITY TEST")
    print("Model:", args.model)
    print("Layer:", args.layer)
    print("Iterations:", args.iterations)
    print("Sentences:", sentence_ids)
    print("============================================\n")

    # --------------------------------------------------------------
    # 1) REPRODUCIBILITY TEST: SAME SEED (or pretrained)
    # --------------------------------------------------------------
    print(">>> TEST 1 : SAME-SEED REPRODUCIBILITY")
    runs = []

    for it in range(args.iterations):
        print(f"  Running iteration {it+1}/{args.iterations}")

        if args.model == "pretrained":
            model = load_model(pretrained=True)
        else:
            if args.seed is None:
                raise ValueError("Untrained model requires --seed")
            model = load_model(pretrained=False, seed=args.seed)

        run_result = {}

        for sid, sent in zip(sentence_ids, sentences):
            bert_tokens = sent
            layer_values = extract_layer_sums(model, tokenizer, bert_tokens, args.layer)
            run_result[sid] = layer_values

        runs.append(run_result)

    print("\n>>> RESULTS (same seed/pretrained):\n")

    # Compare run1 vs run2, run1 vs run3, ...
    for layer in (runs[0][sentence_ids[0]].keys()):
        print(f"Layer {layer}:")
        for i in range(1, args.iterations):
            diffs = []

            for sid in sentence_ids:
                cmp = compare_runs(runs[0][sid], runs[i][sid], args.tolerance)
                diffs.append(cmp[layer]["top5_ok"])

            if all(diffs):
                print(f"   Iteration 1 vs {i+1}: REPRODUCIBLE (top-5 identical)")
            else:
                print(f"   Iteration 1 vs {i+1}: NOT reproducible")


    # --------------------------------------------------------------
    # 2) MULTI-SEED SENSITIVITY TEST  (optional)
    # --------------------------------------------------------------
    if args.model == "untrained" and args.compare_seeds:
        print("\n============================================")
        print(">>> TEST 2 : DIFFERENT-SEED SENSITIVITY")
        print("============================================\n")

        seeds_to_test = [args.seed, args.seed + 1, args.seed + 2]

        seed_runs = {}

        for sd in seeds_to_test:
            print(f"  Running seed {sd}...")
            model = load_model(pretrained=False, seed=sd)

            run_data = {}
            for sid, sent in zip(sentence_ids, sentences):
                bert_tokens = sent
                layer_values = extract_layer_sums(model, tokenizer, bert_tokens, args.layer)
                run_data[sid] = layer_values

            seed_runs[sd] = run_data

        seeds_list = list(seed_runs.keys())

        print("\n>>> RESULTS (different seeds):\n")

        for layer in (seed_runs[seeds_list[0]][sentence_ids[0]].keys()):
            print(f"Layer {layer}:")
            for i in range(len(seeds_list)-1):
                sA = seeds_list[i]
                sB = seeds_list[i+1]

                diffs = []

                for sid in sentence_ids:
                    cmp = compare_runs(seed_runs[sA][sid], seed_runs[sB][sid], args.tolerance)
                    diffs.append(cmp[layer]["top5_ok"])

                if any(diffs):  
                    print(f"   Seeds {sA} vs {sB}: SOME overlap in top5 (expected small)")
                else:
                    print(f"   Seeds {sA} vs {sB}: COMPLETELY different top5 (excellent sensitivity)")

    print("\nDone.\n")


if __name__ == "__main__":
    main()