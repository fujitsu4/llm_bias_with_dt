"""
compute_bert_basic_features.py
Author: Zakaria JOUILIL

Description:
    Compute basic BERT token features:
        - is_special_token
        - token_length
        - is_subword
        - position_in_sentence
        - relative_position
        - num_subtokens

Inputs:
    --bert_csv : Path to bert_tokens.csv

Outputs:
    - CSV file containing all original BERT tokens + new feature columns.
    (Default: outputs/bert/bert_basic_features.csv)

Usage:
    python -m src.bert.compute_bert_basic_features \
        --bert_csv outputs/bert/bert_tokens.csv \
        --output outputs/bert/bert_basic_features.csv
"""

import argparse
import pandas as pd
from src.utils.paths import get_project_path


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------
parser = argparse.ArgumentParser(description="Compute basic BERT token features")
parser.add_argument("--bert_csv", type=str, required=True,
                    help="Input bert_tokens.csv file")
parser.add_argument("--output", type=str, default="outputs/bert/bert_basic_features.csv",
                    help="Output CSV file")

args = parser.parse_args()

bert_csv = get_project_path(*args.bert_csv.split("/"))
output_csv = get_project_path(*args.output.split("/"))

print("[INFO] Loading BERT tokens from:", bert_csv)
df = pd.read_csv(bert_csv, sep=";", keep_default_na=False, na_values=[], dtype={"bert_token": str})

df["sentence_id"] = df["sentence_id"].astype(int)
df["bert_index"] = df["bert_index"].astype(int)
df["bert_token"] = df["bert_token"].fillna("").astype(str)

# ----------------------------------------------------------
# 1) is_special_token
# ----------------------------------------------------------
df["is_special_token"] = df["bert_token"].isin(["[CLS]", "[SEP]", "[PAD]", "[UNK]"]).astype(int)

# ----------------------------------------------------------
# 2) token_length
# ----------------------------------------------------------
df["token_length"] = df["bert_token"].str.len()   # returns 0 for empty strings (CLS/SEP)

# ----------------------------------------------------------
# 3) is_subword
# ----------------------------------------------------------
df["is_subword"] = df["bert_token"].str.startswith("##").fillna(False).astype(int)

# ----------------------------------------------------------
# 4) position_in_sentence   (duplicate of bert_index)
# ----------------------------------------------------------
df["position_in_sentence"] = df["bert_index"]

# ----------------------------------------------------------
# 5) relative_position = bert_index / max_index
# ----------------------------------------------------------
print("[INFO] Computing relative positions...")

df["relative_position"] = df.groupby("sentence_id")["bert_index"].transform(
    lambda col: col / col.max()
)

# ----------------------------------------------------------
# 6) num_subtokens
# ----------------------------------------------------------

# ---- normalize word_id: convert possible floats "3.0" to int; keep None/'' as NaN for grouping ----
def normalize_word_id(x):
    if x is None:
        return pd.NA
    s = str(x).strip()
    if s == "":
        return pd.NA
    try:
        return int(float(s))
    except:
        return pd.NA

df["word_id"] = df["word_id"].apply(normalize_word_id)

print("[INFO] Computing num_subtokens per word...")

# ---- num_subtokens: number of BERT tokens per original word (per sentence_id, word_id) ----
# compute counts for rows that have a valid word_id
counts = (
    df[df["word_id"].notna()]
    .groupby(["sentence_id", "word_id"])
    .size()
    .rename("num_subtokens")
    .reset_index()
)

# merge counts back to df
df = df.merge(counts, how="left", on=["sentence_id", "word_id"])

# set num_subtokens for special tokens / missing word_id to -1
df["num_subtokens"] = df["num_subtokens"].fillna(-1).astype(int)

# ----------------------------------------------------------
# 7) Save output
# ----------------------------------------------------------
df.to_csv(output_csv, sep=";", index=False)
print("[INFO] Saved BERT features to:", output_csv)