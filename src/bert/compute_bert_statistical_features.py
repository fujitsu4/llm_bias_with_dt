"""
compute_bert_statistical_features.py
Author: Zakaria JOUILIL

Description:
    Compute statistical features for BERT tokens based on the enriched
    bert_basic_features.csv file.

    Added features:
        - token_relative_frequency  (BERT-token-level, per sentence)
        - word_sentence_frequency  (spaCy-word-level, global corpus, propagated)
        - token_rank               (BERT-token-level, global corpus)
        - word_burstiness          (spaCy-word-level, global corpus, propagated)

Input:
    --bert_features : outputs/bert/bert_basic_features.csv

Output:
    A CSV containing all original columns + the 4 new statistical features.

Usage :
    python -m src.bert.compute_bert_statistical_features --bert_features outputs/bert/bert_basic_features.csv --output outputs/bert/bert_statistical_features.csv
"""

import argparse
import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def compute_burstiness(counts_per_sentence_dict):
    """
    Given a dict: sentence_id -> count_in_sentence
    Returns burstiness = (σ - μ) / (σ + μ)
    """
    counts = np.array(list(counts_per_sentence_dict.values()), dtype=float)
    if len(counts) == 0:
        return 0.0
    mu = counts.mean()
    sigma = counts.std()
    if mu + sigma == 0:
        return 0.0
    return float((sigma - mu) / (sigma + mu))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Compute statistical BERT features")
parser.add_argument("--bert_features", required=True, help="Input bert_basic_features.csv")
parser.add_argument("--output", required=True, help="Output CSV path")
args = parser.parse_args()

print("[INFO] Loading BERT basic features from:", args.bert_features)
df = pd.read_csv(args.bert_features, sep=";", keep_default_na=False, na_values=[""])

# ------------------------------------------------------------
# Normalization of numeric fields
# ------------------------------------------------------------
df["sentence_id"] = df["sentence_id"].astype(int)
df["bert_index"] = df["bert_index"].astype(int)
df["word_id"] = df["word_id"].replace({"": np.nan}).astype(float)
df["num_subtokens"] = df["num_subtokens"].astype(int)
df["is_special_token"] = df["is_special_token"].astype(int)

# ------------------------------------------------------------
# 1) token_relative_frequency
#    (token count in sentence / total tokens in sentence)
# ------------------------------------------------------------
print("[INFO] Computing token_relative_frequency (per sentence)...")

def relative_freq(group):
    return group.groupby("bert_token")["bert_token"].transform("count") / len(group)

df["token_relative_frequency"] = df.groupby("sentence_id", group_keys=False).apply(relative_freq)

# Special tokens must ALWAYS have local frequency = 0
df.loc[df["is_special_token"] == 1, "token_relative_frequency"] = 0.0

# ------------------------------------------------------------
# 2) word_sentence_frequency (spaCy word level, global corpus)
#    Count number of DISTINCT sentences in which each word_id appears.
#    Propagate to all BERT subtokens with that word_id.
# ------------------------------------------------------------
print("[INFO] Computing word_sentence_frequency (global, spaCy-word-level)...")

# Filter valid words (no specials)
df_words = df[df["word_id"].notna()].copy()

# Count distinct sentences per (sentence_id, word) using reconstructed word form
# Reconstructed word = all subtokens for that word merged
# We must reconstruct to avoid issues with ##

print("[INFO] Reconstructing spaCy words from BERT subtokens...")

def clean_tok(t):
    return t[2:] if t.startswith("##") else t

# Build reconstructed words
df_words_copy = df_words.copy()
df_words_copy["clean_subtoken"] = df_words_copy["bert_token"].apply(clean_tok)
reconstructed = (
    df_words_copy.groupby(["sentence_id", "word_id"])['clean_subtoken']
    .apply(lambda toks: "_".join(toks))
    .reset_index(name="word_form")
)

# Sentence frequency per unique word_form
word_sentfreq = reconstructed.groupby("word_form")["sentence_id"].nunique().rename("word_sentence_frequency")

# Merge back
reconstructed = reconstructed.merge(word_sentfreq, on="word_form", how="left")

# Attach these frequencies back to the full DF by (sentence_id, word_id)
df = df.merge(reconstructed[["sentence_id", "word_id", "word_sentence_frequency"]],
              on=["sentence_id", "word_id"], how="left")

# Specials get 0
df["word_sentence_frequency"] = df["word_sentence_frequency"].fillna(0).astype(int)
df.loc[df["is_special_token"] == 1, "word_sentence_frequency"] = 0

# ------------------------------------------------------------
# 3) token_rank (global frequency rank of BERT tokens)
# ------------------------------------------------------------
print("[INFO] Computing token_rank (BERT-token-level)...")
# Compute token frequencies
token_counts = df[df["is_special_token"] == 0].groupby("bert_token").size()

# Assign tied ranks (same frequency -> same rank)
token_ranks = token_counts.rank(method="min", ascending=False).astype(int)

# Map to DF
df["token_rank"] = df["bert_token"].map(token_ranks)

# Replace all missing ranks with 0
df["token_rank"] = df["token_rank"].fillna(0)

# Specials = 0
df.loc[df["is_special_token"] == 1, "token_rank"] = 0

df["token_rank"] = df["token_rank"].astype(int)

# ------------------------------------------------------------
# 4) word_burstiness (spaCy-word-level)
# ------------------------------------------------------------
print("[INFO] Computing word_burstiness (spaCy-word-level)...")

# Count occurrences per sentence
df_words_copy2 = df_words_copy.copy()  # from above; contains clean subtokens
# Recompute reconstructed words for safety
reconstructed2 = (
    df_words_copy2.groupby(["sentence_id", "word_id"])['clean_subtoken']
    .apply(lambda toks: "_".join(toks))
    .reset_index(name="word_form")
)

# Count occurrences per sentence (word_form)
word_sentence_counts = reconstructed2.groupby(["word_form", "sentence_id"]).size().rename("count")

# Compute burstiness for each word_forms
burstiness_values = {}
for word, subdf in word_sentence_counts.groupby(level=0):
    # Build dict: sentence_id -> count
    counts_dict = {sid: cnt for (_, sid), cnt in subdf.items()}
    burstiness_values[word] = compute_burstiness(counts_dict)

burstiness_df = pd.DataFrame(
    [(w, burstiness_values[w]) for w in burstiness_values],
    columns=["word_form", "word_burstiness"]
)

# Merge burstiness back to reconstructed
reconstructed2 = reconstructed2.merge(burstiness_df, on="word_form", how="left")

# Merge burstiness into df
df = df.merge(reconstructed2[["sentence_id", "word_id", "word_burstiness"]],
              on=["sentence_id", "word_id"], how="left")

# Specials get 0
df["word_burstiness"] = df["word_burstiness"].fillna(0.0)

# ------------------------------------------------------------
# Final column order
# ------------------------------------------------------------
final_cols = [
    "sentence_id", "bert_index", "bert_token", "word_id", "is_special_token",
    "token_length", "is_subword", "position_in_sentence", "relative_position",
    "num_subtokens", "token_relative_frequency", "word_sentence_frequency",
    "token_rank", "word_burstiness", "dataset"
]

df = df[final_cols]

# ------------------------------------------------------------
# Save output
# ------------------------------------------------------------
print("[INFO] Saving output to:", args.output)
df.to_csv(args.output, sep=";", index=False)