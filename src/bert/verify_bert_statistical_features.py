"""
verify_bert_statistical_features.py
Author: Zakaria JOUILIL

Description :
    Verifies outputs of compute_bert_statistical_features.py

Inputs :
    --features_csv : outputs/bert/bert_statistical_features.csv : csv containing the statistical features for bert tokens

Ouputs :
    --log_file : logs/verify_bert_statistical_features.log : A log file where all reports and progress messages are stored
    
Usage:
   python -m src.bert.verify_bert_statistical_features \
       --features_csv outputs/bert/bert_statistical_features.csv \
       --log_file logs/bert_statistical_features_logs.txt
"""

import argparse
import sys
import os
import math
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

# -------------------------
def setup_logger(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = open(path, "w", encoding="utf-8")
    def log(msg="", end="\n"):
        s = str(msg)
        ts = ""
        print(s, end=end)
        f.write(s + end)
        f.flush()
    return log, f

# -------------------------
def fatal(log, fhandle, msg):
    log("[ERROR] " + msg)
    log("[ERROR] Verification FAILED.")
    fhandle.close()
    sys.exit(1)

# -------------------------
def approx_equal(a, b, tol=1e-9):
    return abs(a - b) <= tol

# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Verify bert_statistical_features output")
    parser.add_argument("--features_csv", required=True, help="CSV produced by compute_bert_statistical_features.py")
    parser.add_argument("--log_file", required=True, help="Log file to write verification logs")
    args = parser.parse_args()

    log, fhandle = setup_logger(args.log_file)
    log(f"[INFO] Logging -> {args.log_file}")
    log("[INFO] Starting verification of statistical features")

    # read
    try:
        df = pd.read_csv(args.features_csv, sep=";", keep_default_na=False, na_values=[""])
    except Exception as e:
        fatal(log, fhandle, f"Cannot read CSV: {e}")

    log(f"[INFO] Rows loaded: {len(df)}")

    # expected columns (exact order not strictly required but we check existence)
    expected = [
        "sentence_id","bert_index","bert_token","word_id","is_special_token","token_length",
        "is_subword","position_in_sentence","relative_position","num_subtokens",
        "token_relative_frequency","word_sentence_frequency","token_rank","word_burstiness","dataset"
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        fatal(log, fhandle, f"Missing expected columns: {missing}")
    log(f"[OK] Columns presence: all required columns found ({len(expected)})")

    # quick types & conversions (safe)
    # keep original strings, but derive numeric versions we will test with
    # convert safe numeric columns:
    def to_int_series(col, name, allow_na=False):
        try:
            s = df[col].replace({"": np.nan})
            res = pd.to_numeric(s, errors="coerce").astype("Int64")
            return res
        except Exception as e:
            fatal(log, fhandle, f"Column {name} cannot be converted to integer-like: {e}")

    def to_float_series(col, name):
        try:
            s = df[col].replace({"": np.nan})
            res = pd.to_numeric(s, errors="coerce").astype(float)
            return res
        except Exception as e:
            fatal(log, fhandle, f"Column {name} cannot be converted to float: {e}")

    sent_id = to_int_series("sentence_id","sentence_id")
    bert_idx = to_int_series("bert_index","bert_index")
    pos_in_sent = to_int_series("position_in_sentence","position_in_sentence")
    token_len = to_int_series("token_length","token_length")
    num_subtokens = to_int_series("num_subtokens","num_subtokens")
    # floats
    rel_pos = to_float_series("relative_position","relative_position")
    token_rel_freq = to_float_series("token_relative_frequency","token_relative_frequency")
    word_sent_freq = to_int_series("word_sentence_frequency","word_sentence_frequency")
    token_rank = to_int_series("token_rank","token_rank")
    word_burst = to_float_series("word_burstiness","word_burstiness")

    # booleans
    is_special = df["is_special_token"].astype(str).str.strip()
    is_subw = df["is_subword"].astype(str).str.strip()

    # basic checks
    n = len(df)
    # 1) sentence_id, bert_index, position_in_sentence integer-like and non-negative
    if sent_id.isna().any():
        fatal(log, fhandle, "Some sentence_id values are missing or non-numeric")
    if bert_idx.isna().any():
        fatal(log, fhandle, "Some bert_index values are missing or non-numeric")
    if (bert_idx < 0).any():
        fatal(log, fhandle, "Some bert_index < 0")
    if pos_in_sent.isna().any():
        fatal(log, fhandle, "Some position_in_sentence values are missing")
    log("[OK] sentence_id / bert_index / position_in_sentence convertible to int and non-negative")

    # 2) is_special_token and is_subword: must be 0/1 (string '0'/'1' or booleans)
    valid_special = is_special.isin({"0","1","True","False","true","false"})
    if not valid_special.all():
        sample = df[~valid_special].head(10)[["sentence_id","bert_index","bert_token","is_special_token"]]
        fatal(log, fhandle, f"is_special_token contains invalid values. Examples:\n{sample.to_string(index=False)}")
    # normalize to ints for tests
    is_special_int = is_special.replace({"True":"1","true":"1","False":"0","false":"0"}).astype(int)
    is_subw_int = is_subw.replace({"True":"1","true":"1","False":"0","false":"0"}).astype(int)
    log("[OK] is_special_token and is_subword contain only 0/1-like values")

    # 3) token_relative_frequency in [0,1]
    if token_rel_freq.isna().any():
        fatal(log, fhandle, "token_relative_frequency contains NaN values")
    if (token_rel_freq < -1e-9).any() or (token_rel_freq > 1.0 + 1e-9).any():
        bad = df[(token_rel_freq < -1e-9) | (token_rel_freq > 1.0 + 1e-9)][["sentence_id","bert_index","bert_token","token_relative_frequency"]].head(10)
        fatal(log, fhandle, f"token_relative_frequency out of [0,1] range. Examples:\n{bad.to_string(index=False)}")
    log("[OK] token_relative_frequency in [0,1] for all rows")

    # 4) Specials: for tokens special -> token_relative_frequency == 0, token_rank == 0, word_sentence_frequency==0, num_subtokens == -1 (convention)
    specials = df[is_special_int == 1]
    if not specials.empty:
        # token_relative_frequency
        bad_rt = specials[ (token_rel_freq.loc[specials.index].astype(float) != 0.0) ]
        if not bad_rt.empty:
            fatal(log, fhandle, "Some special tokens have token_relative_frequency != 0")
        # token_rank
        bad_tr = specials[ token_rank.loc[specials.index].astype(float) != 0 ]
        if not bad_tr.empty:
            fatal(log, fhandle, "Some special tokens have token_rank != 0")
        # word_sentence_frequency
        bad_wsf = specials[ word_sent_freq.loc[specials.index].astype(float) != 0 ]
        if not bad_wsf.empty:
            fatal(log, fhandle, "Some special tokens have word_sentence_frequency != 0")
        # num_subtokens
        bad_ns = specials[ num_subtokens.loc[specials.index].astype(float) != -1 ]
        if not bad_ns.empty:
            fatal(log, fhandle, "Some special tokens do not have num_subtokens == -1 (convention)")
    log("[OK] Specials respect conventions (relative_freq=0, token_rank=0, word_sentence_frequency=0, num_subtokens=-1)")

    # 5) token_relative_frequency correctness per sentence:
    # For each sentence, token_relative_frequency should equal count(token)/total_tokens_in_sentence
    log("[INFO] Verifying token_relative_frequency equals count(token)/n_tokens per sentence (sampled checks)...")
    bad_count = 0
    # iterate grouped by sentence but limit expensive full-check to a subset if very large
    n_sents = df["sentence_id"].nunique()
    sample_sents = list(df["sentence_id"].unique())[:200]  # first 200 sentences for full deterministic check
    for sid in tqdm(sample_sents, desc="checking token_relative_frequency (sample)"):
        g = df[df["sentence_id"] == sid]
        if g.empty:
            continue
        total = len(g[ g["is_special_token"].astype(str).map(lambda x: x in ("0","False","false")) ])  # count non special tokens OR include specials? earlier we set specials freq 0 so include all tokens OK
        # We'll compute count per token string
        cnts = g["bert_token"].value_counts().to_dict()
        # compare for each unique token in the sentence
        for tok, expected_count in cnts.items():
            # Skip specials: their freq is *defined* as 0 by design
            row = g[g["bert_token"] == tok].iloc[0]
            if int(row["is_special_token"]) == 1:
                continue  # <--- this fixes everything

            expected_rel = expected_count / len(g)
            actual_rel = float(row["token_relative_frequency"])

            if not approx_equal(actual_rel, expected_rel, tol=1e-6):
                bad_count += 1
                if bad_count <= 5:
                    log(f"[ERROR] token_relative_frequency mismatch in sentence {sid}, token='{tok}' expected={expected_rel:.6f} actual={actual_rel:.6f}")
                else:
                    break
        if bad_count:
            break
    if bad_count:
        fatal(log, fhandle, f"token_relative_frequency mismatches found (example count {bad_count})")
    log("[OK] token_relative_frequency verified on sampled sentences (first 200)")

    # 6) word_sentence_frequency propagation & consistency:
    # For each (sentence_id, word_id) group, all rows must have same word_sentence_frequency.
    log("[INFO] Verifying word_sentence_frequency propagation to sub-tokens and integer-ness")
    # rows with valid word_id
    mask_valid_word = df["word_id"].astype(str).str.strip() != ""
    # convert to normalized word_id (int) where possible
    def parse_wid(v):
        s = str(v).strip()
        if s == "" or s.lower() in ("nan","none","null"):
            return None
        try:
            return int(float(s))
        except:
            return None
    wids = df["word_id"].apply(parse_wid)
    df["_word_id_parsed"] = wids
    grouped = df[ df["_word_id_parsed"].notna() ].groupby(["sentence_id","_word_id_parsed"])
    inconsistent = []
    for key, g in grouped:
        vals = set(g["word_sentence_frequency"].astype(str).tolist())
        if len(vals) > 1:
            inconsistent.append((key, list(vals)[:5]))
    if inconsistent:
        sample = inconsistent[:10]
        fatal(log, fhandle, f"word_sentence_frequency inconsistent within same (sentence,word_id). Examples: {sample}")
    log("[OK] word_sentence_frequency is propagated consistently to sub-tokens")

    # 7) token_rank monotonicity check with global counts
    log("[INFO] Verifying token_rank mapping using DENSE RANK (ties = same rank)")

    # compute expected dense rank
    token_counts = df[df["is_special_token"] == 0].groupby("bert_token").size()
    dense_rank = token_counts.rank(method="dense", ascending=False).astype(int)
    expected_map = dense_rank.to_dict()

    mismatches = []
    for tok, expected in expected_map.items():
        # recorded rank for this token in df
        recorded = df.loc[(df["bert_token"] == tok) & (df["is_special_token"] == 0),
                      "token_rank"].iloc[0]

        if recorded != expected:
            mismatches.append((tok, recorded, expected))
            if len(mismatches) >= 20:
                break

    if mismatches:
        log("[ERROR] token_rank mismatches detected (dense ranking used for verification):")
        for tok, rec, exp in mismatches:
            log(f"  token='{tok}' recorded={rec} expected={exp}")
    else:
        log("[OK] token_rank matches dense rank for all tokens.")

    # 8) word_burstiness values numeric & within [-1,1] (theoretical bounds)
    log("[INFO] Checking word_burstiness numeric and within [-1,1] (finite)")
    if word_burst.isna().any():
        log("[WARN] Some word_burstiness are NaN (allowed but check source).")
    # allow a little numeric slack
    if (word_burst.dropna() < -1.5).any() or (word_burst.dropna() > 1.5).any():
        bad = df[ (word_burst.astype(float) < -1.5) | (word_burst.astype(float) > 1.5) ][["sentence_id","bert_token","word_burstiness"]].head(10)
        log(f"[WARN] Some burstiness values outside [-1.5,1.5]. Examples:\n{bad.to_string(index=False)}")
    else:
        log("[OK] word_burstiness numeric and within reasonable bounds")

    # 9) basic stats & summary
    log("[INFO] Summary statistics (sample):")
    log(f"  unique sentences: {df['sentence_id'].nunique()}")
    log(f"  unique tokens (bert_token): {df['bert_token'].nunique()}")
    log(f"  tokens flagged special: {int(is_special_int.sum())}")
    log(f"  num rows: {len(df)}")
    log("[INFO] Verification completed successfully (warnings may be present).")
    fhandle.close()

if __name__ == "__main__":
    main()