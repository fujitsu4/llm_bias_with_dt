"""
merge_spacy_and_bert.py
Author: Zakaria JOUILIL

Description:
    Merge spaCy features (word-level) with BERT basic features (token-level) and
    statistical features (token or word level) into a unified dataframe that will
    later be enriched with neighbor POS features.

Final column order:
    sentence_id
    bert_index
    bert_token
    word_id
    is_numeric
    is_punctuation
    is_stopword
    is_pronoun
    pos_id
    dep_id
    ent_id
    depth_in_tree
    num_dependents
    is_special_token
    token_length
    is_subword
    position_in_sentence
    relative_position
    num_subtokens
    token_relative_frequency
    word_sentence_frequency
    token_rank
    word_burstiness
    prev_pos_id       (placeholder = -1)
    next_pos_id       (placeholder = -1)
    dataset

Inputs :
    --spacy_csv : outputs/spacy/spacy_features.csv (spacy features)
    --bert_csv : outputs/bert/bert_statistical_features.csv (basic AND statistical bert features)

Output :
    outputs/merged/spacy_bert_merged.csv (a csv containing spacy and bert features)

Usage:
    python -m src.bert.merge_spacy_and_bert \
        --spacy_csv outputs/spacy/spacy_features.csv \
        --bert_csv outputs/bert/bert_statistical_features.csv \
        --output outputs/bert/spacy_bert_merged.csv

A brief Note :
    We will keep only the numeric columns that can be directly used as inputs for
    our decision tree model. The following columns will therefore be dropped since
    they were only useful for debugging.
        - pos
        - dep_label
        - ent_label
"""

import argparse
import pandas as pd

# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Merge spaCy and BERT features")
parser.add_argument("--spacy_csv", required=True, help="Path to spacy_features.csv")
parser.add_argument("--bert_csv", required=True, help="Path to bert_statistical_features.csv")
parser.add_argument("--output", required=True, help="Output merged CSV path")

args = parser.parse_args()

print("[INFO] Loading spaCy features:", args.spacy_csv)
spacy_df = pd.read_csv(args.spacy_csv, sep=";", keep_default_na=False, na_values=[""])

print("[INFO] Loading BERT basic AND statistical features:", args.bert_csv)
bert_df = pd.read_csv(args.bert_csv, sep=";", keep_default_na=False, na_values=[""])

# -------------------------------------------------------------------
# 1. Normalize column names for merging
# -------------------------------------------------------------------
spacy_df = spacy_df.rename(columns={"word_index": "word_id"})

# -------------------------------------------------------------------
# 2. Drop unused STRING columns (only keep IDs for ML)
# -------------------------------------------------------------------
cols_to_drop = ["pos", "dep_label", "ent_label"]
for col in cols_to_drop:
    if col in spacy_df.columns:
        spacy_df = spacy_df.drop(columns=[col])

# -------------------------------------------------------------------
# 3. Merge on (sentence_id, word_id)
# -------------------------------------------------------------------
print("[INFO] Merging on (sentence_id, word_id)...")
merged = bert_df.merge(
    spacy_df,
    on=["sentence_id", "word_id", "dataset"],
    how="left"
)

# -------------------------------------------------------------------
# 4. Initialize prev/next POS IDs (will be computed in next script)
# -------------------------------------------------------------------
merged["prev_pos_id"] = -1
merged["next_pos_id"] = -1

# -------------------------------------------------------------------
# 5. Reorder columns
# -------------------------------------------------------------------
final_cols = [
    "sentence_id",
    "bert_index",
    "bert_token",
    "word_id",               #word_index renamed

    "is_numeric",
    "is_punctuation",
    "is_stopword",
    "is_pronoun",
    "pos_id",
    "dep_id",
    "ent_id",
    "depth_in_tree",
    "num_dependents",

    "is_special_token",
    "token_length",
    "is_subword",
    "position_in_sentence",
    "relative_position",
    "num_subtokens",
    "token_relative_frequency",
    "word_sentence_frequency",
    "token_rank",
    "word_burstiness",

    "prev_pos_id",
    "next_pos_id",

    "dataset"
]

print("[INFO] Reordering columns...")
merged = merged[final_cols]

# -------------------------------------------------------------------
# 6. Assign correct default values for special tokens
# -------------------------------------------------------------------
print("[INFO] Assign defautl values for special tokens...")

merged["word_id"] = merged["word_id"].fillna(-1).astype(int)

special_mask = (merged["is_special_token"] == 1)

# 1) Boolean spaCy features → 0
bool_cols = ["is_numeric", "is_punctuation", "is_stopword", "is_pronoun"]
for col in bool_cols:
    if col in merged.columns:
        merged.loc[special_mask, col] = 0

# 2) Syntactic or contextual features → -1 (the value 0 has already a real meaning)
id_cols_minus1 = ["pos_id", "dep_id", "ent_id", "depth_in_tree", "num_dependents"]
for col in id_cols_minus1:
    if col in merged.columns:
        merged.loc[special_mask, col] = -1

# -------------------------------------------------------------------
# 7. Save output
# -------------------------------------------------------------------
print("[INFO] Saving merged output to:", args.output)
merged.to_csv(args.output, sep=";", index=False)

print("[INFO] Merge completed successfully.")