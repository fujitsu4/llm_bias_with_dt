"""
compute_neighbor_pos_features.py
Author: Zakaria JOUILIL

Description:
    Compute spaCy neighbor POS features for merged BERT+spaCy tokens.
    Specifically fills prev_pos_id and next_pos_id using the spaCy pos_id.

    HOW IT WORKS:
      - For each sentence_id, we sort tokens by word_id.
      - For each token, prev_pos_id = pos_id of word_id - 1
                           next_pos_id = pos_id of word_id + 1
      - Tokens with irrelevant word_id (including specials):
            prev_pos_id = -1
            next_pos_id = -1

Input:
    --input    : merged CSV (output of merge_spacy_and_bert.py)
Output:
    --output   : same CSV + filled prev_pos_id, next_pos_id

Usage:
    python -m src.bert.compute_neighbor_pos_features \
        --input outputs/bert/spacy_bert_merged.csv \
        --output outputs/bert/bert_final_features.csv
"""

import argparse
import pandas as pd
import numpy as np

# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Compute neighbor POS features")
parser.add_argument("--input", required=True, help="Merged input CSV")
parser.add_argument("--output", required=True, help="Output CSV path")
args = parser.parse_args()

print("[INFO] Loading merged file:", args.input)
df = pd.read_csv(args.input, sep=";", keep_default_na=False, na_values=[""])

# -------------------------------------------------------------------
# Sanity checks
# -------------------------------------------------------------------
required_cols = {
    "sentence_id", "word_id", "pos_id",
    "prev_pos_id", "next_pos_id",
    "is_special_token"
}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"[ERROR] Missing required columns in input: {missing}")

# -------------------------------------------------------------------
# Initialize outputs
# -------------------------------------------------------------------
print("[INFO] Computing neighbor POS features...")

prev_pos = np.full(len(df), -1, dtype=int)
next_pos = np.full(len(df), -1, dtype=int)

# -------------------------------------------------------------------
# Process per sentence
# -------------------------------------------------------------------
for sid, group in df.groupby("sentence_id"):
    idx = group.index

    # keep ONLY real words
    valid_rows = group[group["is_special_token"] == 0]
    if valid_rows.empty:
        continue

    valid_sorted = valid_rows.sort_values("word_id")

    word_ids = valid_sorted["word_id"].astype(int).values
    pos_ids = valid_sorted["pos_id"].astype(int).values
    original_idx = valid_sorted.index.values

    mapping = dict(zip(word_ids, pos_ids))

    for wid, row_index in zip(word_ids, original_idx):
        prev_pos[row_index] = mapping.get(wid - 1, -1)
        next_pos[row_index] = mapping.get(wid + 1, -1)

# Final overwrite for special tokens
df.loc[df["is_special_token"] == 1, ["prev_pos_id", "next_pos_id"]] = -1

df["prev_pos_id"] = prev_pos
df["next_pos_id"] = next_pos

# -------------------------------------------------------------------
# Save
# -------------------------------------------------------------------
print("[INFO] Saving with neighbor features to ", args.output)
df.to_csv(args.output, sep=";", index=False)

print("[INFO] Neighbor POS computation completed successfully.")