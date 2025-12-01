"""
compute_layer_attention_sums.py
Author: Zakaria JOUILIL

Description:
    Extract attention matrices from a BERT model (pretrained or randomly initialized),
    aggregate attention per-layer and produce per-token layer-sums and top-k flags.

Outputs:
    - CSV with the same rows as input (one row per bert token) plus:
        layer_1_sum, ..., layer_12_sum  (floats)
        top5_l1, ..., top5_l12          (0/1 flags whether token is top-k in sentence for that layer)

Usage examples:
    # pretrained
    python -m src.attention.compute_layer_attention_sums \
        --model pretrained \
        --input_csv outputs/bert/bert_final_features.csv \
        
    # untrained with seed
    python -m src.attention.compute_layer_attention_sums \
        --model untrained --seed 100 \
        --input_csv outputs/bert/bert_final_features.csv \
"""

import argparse
import os
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel, BertConfig
import math

def fix_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except:
        pass

    # deterministic flags (may affect perf)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(pretrained: bool, device):
    if pretrained:
        # load pretrained weights
        model = BertModel.from_pretrained("bert-base-cased")
    else:
        # load config from pretrained to keep same architecture, but random init
        cfg = BertConfig.from_pretrained("bert-base-cased")
        model = BertModel(cfg)  # random init
    model.config.output_attentions = True
    model.to(device)
    model.eval()
    return model

def aggregate_attention_per_layer(attentions):
    """
    attentions: tuple of length L where each element is tensor (batch, n_heads, seq_len, seq_len)
    returns: list of arrays length seq_len with aggregated sums per layer
             (list length = number_of_layers; each item a numpy array shape (seq_len,))
    """
    layer_sums = []
    # attentions is tuple(layer0, layer1, ..., layerN-1)
    for layer_att in attentions:
        # layer_att: (batch=1, n_heads, seq_len, seq_len)
        # we'll work on batch[0]
        arr = layer_att[0].detach().cpu().numpy()  # shape (n_heads, seq_len, seq_len)
        # sum across heads first -> shape (seq_len, seq_len)
        sum_heads = np.sum(arr, axis=0)  # (seq_len, seq_len)
        # total attention received by token j: sum over source i of sum_heads[i, j]
        per_token = np.sum(sum_heads, axis=0)  # sum over source dimension -> shape (seq_len,)
        layer_sums.append(per_token)
    return layer_sums

def topk_flags(arr, k=5):
    """
    arr: 1D numpy array
    returns boolean array same shape True for elements in top-k.
    - ties are resolved by selecting topk indices by descending value; if many tied values create ambiguity,
      we use numpy.argsort which gives a deterministic ordering.
    """
    seq_len = arr.shape[0]
    kk = min(k, seq_len)
    # argsort descending
    idx_desc = np.argsort(-arr)
    topk_idx = idx_desc[:kk]
    flags = np.zeros(seq_len, dtype=int)
    flags[topk_idx] = 1
    return flags

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["pretrained", "untrained"], required=True,
                        help="pretrained or untrained model")
    parser.add_argument("--seed", type=int, default=42, help="seed for untrained model")
    parser.add_argument("--input_csv", required=True, help="CSV with bert tokens and features (one row per token)")
    parser.add_argument("--top_k", type=int, default=5, help="k for top-k labeling (default 5)")
    parser.add_argument("--device", type=str, default=None, help="torch device to use (cpu or cuda). autodetect if omitted")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model == "pretrained":
        output_dir = "outputs/attention/pretrained"
        
    else:  # untrained
        if args.seed is None:
            raise ValueError("For untrained model, you MUST provide --seed")
        fix_seed(args.seed)
        output_dir = "outputs/attention/untrained"
        
    print(f"[INFO] Loading tokenizer & model ({args.model}) on {device} ...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
    model = load_model(args.model=="pretrained", device)
    model.eval()

    print(f"[INFO] Loading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv, sep=";", keep_default_na=False)
    # required minimal columns
    for col in ("sentence_id","bert_index","bert_token"):
        if col not in df.columns:
            raise ValueError(f"[ERROR] input CSV must contain column: {col}")

    # prepare output columns
    n_layers = model.config.num_hidden_layers  # typically 12
    # Pre-allocate columns with None / zeros
    for L in range(1, n_layers+1):
        df[f"layer_{L}_sum"] = np.nan
        df[f"top5_l{L}"] = 0

    os.makedirs(output_dir, exist_ok=True)
    if args.model == "pretrained":
        out_name = f"attention_pretrained_top{args.top_k}.csv"
    else:
        out_name = f"attention_untrained_seed_{args.seed}_top{args.top_k}.csv"
    out_path = os.path.join(output_dir, out_name)

    # Process per sentence
    grouped = df.groupby("sentence_id", sort=False)
    print(f"[INFO] Processing {len(grouped)} sentences ...")

    # We'll use convert_tokens_to_ids to ensure exact tokens feed to model (no re-tokenization)
    # For safety, we will ensure tokens list contains the exact special tokens if present in CSV.

    for sid, group in tqdm(grouped, total=len(grouped), desc="sentences"):
        # group is ordered as in CSV; ensure sorted by bert_index
        group = group.sort_values("bert_index")
        idxs = group.index.tolist()
        tokens = group["bert_token"].astype(str).tolist()

        # convert tokens to ids (preserves exact tokens)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        # If any token had an unknown id, convert_tokens_to_ids returns tokenizer.unk_token_id
        # Build tensors
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, device=device)

        with torch.no_grad():
            # ask for attentions explicitly
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
            attentions = outputs.attentions  # tuple length n_layers, each (batch, n_heads, seq_len, seq_len)

        # aggregate per layer
        per_layer_sums = aggregate_attention_per_layer(attentions)  # list length n_layers; arrays shape (seq_len,)

        seq_len = len(tokens)
        # write per-layer sums into df rows
        for li, arr in enumerate(per_layer_sums):
            # arr shape (seq_len,)
            # convert to python floats for each token row
            colname = f"layer_{li+1}_sum"
            for j, df_index in enumerate(idxs):
                # store scalar
                df.at[df_index, colname] = float(arr[j])

            # compute topk flags inside this sentence (based on arr)
            flags = topk_flags(arr, k=args.top_k)
            topcol = f"top5_l{li+1}"
            for j, df_index in enumerate(idxs):
                df.at[df_index, topcol] = int(flags[j])

    # Final touches: ensure numeric types for sums and flags are correct
    for L in range(1, n_layers+1):
        df[f"layer_{L}_sum"] = df[f"layer_{L}_sum"].astype(float)
        df[f"top5_l{L}"] = df[f"top5_l{L}"].astype(int)

    # Save
    print(f"[INFO] Writing output to: {out_path} (this may be large)...")
    df.to_csv(out_path, sep=";", index=False)
    print("[OK] Done. Output written.")

if __name__ == "__main__":
    main()
