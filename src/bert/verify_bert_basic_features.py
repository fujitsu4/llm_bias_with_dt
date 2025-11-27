""" verify_bert_basic_features.py
Author: Zakaria JOUILIL

Description:
   Exhaustive verification of the output of compute_bert_basic_features.py
   Checks performed (non-exhaustive list):
     - expected columns + types
     - is_special_token matches token string
     - is_subword matches token string (## prefix)
     - token_length matches length of token string (including "##")
     - position_in_sentence equals bert_index
     - relative_position == bert_index / max_index (per sentence) within tolerance
     - num_subtokens consistency: for each (sentence_id, word_id) group, count
         tokens and compare to provided num_subtokens; specials -> -1
     - num_subtokens are integers, non-negative (or -1 for specials)
     - dataset column present and last (not required but checked)
     - basic statistics and warnings for suspicious values (very large num_subtokens)

 Usage:
    !python -m src.bert.verify_bert_basic_features --features_csv outputs/bert/bert_basic_features.csv \
      --log_file logs/bert_basic_features_logs.txt
"""
"""
-----------------------------------------------------------------------------
Notes / conventions
-----------------------------------------------------------------------------
 num_subtokens design:
 - num_subtokens is defined at the WORD level and copied to every BERT
   sub-token that belongs to the same original word (word_id).
   Example: sentence "I am playing" -> tokens: [CLS] I am play ##ing [SEP]
     word_id:    None 0  1    2      2    None
     num_subtokens for word_id=2 = 2
     thus both "play" and "##ing" have num_subtokens = 2.
     (is_subword for "play" = 0 and for "##ing" = 1)

 - Special tokens ([CLS],[SEP],[PAD],[UNK]) are assigned num_subtokens = -1
   (marker that the token is not associated with an original word).

 - This convention (num_subtokens = number of BERT pieces for the *word*)
   follows common practice in BERT token-level analyses and is used later in
   downstream merges and decision-tree features.

 Rationale:
 - num_subtokens describes how many BERT pieces the original word was split
   into. It is not "the number of sub-tokens after this piece" but a property
   of the whole word. This makes aggregation and comparisons consistent.
-----------------------------------------------------------------------------
"""

import argparse
import sys
import os
import math
from collections import Counter
import pandas as pd
import numpy as np
from datetime import datetime

def setup_logger(path):
    # simple logger: write console + file
    logfile = open(path, "w", encoding="utf-8")
    def log(msg, end="\n"):
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        out = f"[{ts}] {msg}"
        print(out, end=end)
        logfile.write(out + ("" if end=="" else "\n"))
        logfile.flush()
    return log, logfile

def fatal(log, msg):
    log(f"[ERROR] {msg}")
    log("[ERROR] Verification FAILED.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Verify BERT basic features CSV")
    parser.add_argument("--features_csv", required=True, help="CSV produced by compute_bert_basic_features.py")
    parser.add_argument("--log_file", required=True, help="Log file path")
    parser.add_argument("--max_subtoken_warn", type=int, default=50,
                        help="Warn if num_subtokens > this threshold (default 50)")
    parser.add_argument("--relative_tol", type=float, default=1e-6,
                        help="Tolerance when checking relative positions")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
    log, logfile_handle = setup_logger(args.log_file)
    log("[INFO] Starting verification of BERT basic features")
    log(f"[INFO] Input CSV: {args.features_csv}")

    # --------- Load CSV ----------
    try:
        df = pd.read_csv(args.features_csv, sep=";", keep_default_na=False, na_values=[""], dtype=str)
    except Exception as e:
        fatal(log, f"Cannot read CSV: {e}")

    log(f"[INFO] Rows loaded: {len(df)}")
    if len(df) == 0:
        fatal(log, "Empty CSV")

    # normalize column names (strip)
    df.columns = [c.strip() for c in df.columns]

    # expected columns (based on compute_bert_basic_features)
    expected_cols = [
        "sentence_id", "bert_index", "bert_token", "word_id",
        "is_special_token", "token_length", "is_subword",
        "position_in_sentence", "relative_position", "num_subtokens", "dataset"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        fatal(log, f"Missing required columns: {missing}")
    log("[INFO] ✓ Required columns present")

    # ensure dataset is last column (not required but preferred)
    if df.columns[-1] != "dataset":
        log("[WARN] 'dataset' column is not the last column; reordering would be cosmetic")

    # ---------- Type conversions ----------
    # We keep original strings but convert numeric fields carefully and record errors
    # Convert sentence_id and bert_index to int
    int_cols = ["sentence_id", "bert_index", "position_in_sentence"]
    converted = {}
    for col in int_cols:
        try:
            converted[col] = df[col].astype(float).astype("Int64")
        except Exception as e:
            fatal(log, f"Column {col} could not be converted to integer-like values: {e}")
    # tokens stay as strings
    tokens = df["bert_token"].fillna("").astype(str)

    # boolean-like columns: is_special_token, is_subword -> must be 0/1
    for col in ["is_special_token", "is_subword"]:
        # allow "0"/"1" or numeric 0/1, or True/False
        vals = df[col].fillna("").astype(str).str.strip()
        invalid = vals[~vals.isin({"0","1","True","False","true","false"})]
        if len(invalid) > 0:
            fatal(log, f"Column {col} contains invalid values (expected 0/1 or True/False). Examples: {invalid.unique()[:10]}")
    # normalize to 0/1 integers
    is_special = df["is_special_token"].astype(str).str.strip().replace({"True":"1","true":"1","False":"0","false":"0"}).astype(int)
    is_subword = df["is_subword"].astype(str).str.strip().replace({"True":"1","true":"1","False":"0","false":"0"}).astype(int)

    # token_length numeric
    try:
        token_length_col = df["token_length"].astype(float).astype("Int64")
    except Exception as e:
        fatal(log, f"token_length column cannot be converted to integers: {e}")

    # relative_position numeric float
    try:
        rel_pos = df["relative_position"].astype(float)
    except Exception as e:
        fatal(log, f"relative_position cannot be converted to float: {e}")

    # num_subtokens numeric (allow negative -1 for specials)
    try:
        num_subtokens = df["num_subtokens"].astype(float)
    except Exception as e:
        fatal(log, f"num_subtokens cannot be converted to numeric: {e}")

    # word_id can be NA - parse to integer-like or pd.NA
    def parse_word_id(x):
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "null"):
            return pd.NA
        try:
            return int(float(s))
        except:
            return pd.NA
    word_id_parsed = df["word_id"].apply(parse_word_id)

    # basic checks
    log("[INFO] Running per-row consistency checks (sampled feedback will be shown)")

    n = len(df)
    sample_n = min(5, n)

    # 1) check is_special_token consistency against token strings
    special_tokens = {"[CLS]","[SEP]","[PAD]","[UNK]"}
    mask_special_by_token = tokens.isin(list(special_tokens)).astype(int)
    mismatch_special = (mask_special_by_token != is_special)
    if mismatch_special.any():
        sample_idx = df.index[mismatch_special][:10].tolist()
        log(f"[ERROR] is_special_token mismatch for {mismatch_special.sum()} rows. Examples index: {sample_idx}")
        for i in sample_idx[:5]:
            log(f"  idx={i} token='{tokens.iat[i]}' is_special_token={is_special.iat[i]}")
        fatal(log, "is_special_token misalignment")

    log(f"[OK] ✓ is_special_token matches token strings for all rows ({n} rows)")

    # 2) check is_subword matches prefix '##'
    mask_subword_by_token = tokens.str.startswith("##").astype(int)
    mismatch_sub = (mask_subword_by_token != is_subword)
    if mismatch_sub.any():
        sample_idx = df.index[mismatch_sub][:10].tolist()
        log(f"[ERROR] is_subword mismatch for {mismatch_sub.sum()} rows. Examples index: {sample_idx}")
        for i in sample_idx[:5]:
            log(f"  idx={i} token='{tokens.iat[i]}' is_subword={is_subword.iat[i]}")
        fatal(log, "is_subword misalignment")
    log(f"[OK] ✓ is_subword consistent with '##' prefix")

    # 3) token_length correctness (note: token_length includes '##' characters)
    calc_len = tokens.str.len().astype(int)
    mismatch_len = (calc_len != token_length_col.astype(int))
    if mismatch_len.any():
        sample_idx = df.index[mismatch_len][:10].tolist()
        log(f"[ERROR] token_length mismatch for {mismatch_len.sum()} rows. Examples index: {sample_idx}")
        for i in sample_idx[:10]:
            log(f"  idx={i} token='{tokens.iat[i]}' len={calc_len.iat[i]} token_length_col={token_length_col.iat[i]}")
        fatal(log, "token_length mismatches found")
    log("[OK] ✓ token_length matches string length for all tokens")

    # 4) position_in_sentence equals bert_index
    pos_mismatch = (converted["bert_index"] != converted["position_in_sentence"])
    if pos_mismatch.any():
        sample_idx = df.index[pos_mismatch][:10].tolist()
        log(f"[ERROR] position_in_sentence != bert_index for {pos_mismatch.sum()} rows. Examples: {sample_idx}")
        fatal(log, "position_in_sentence mismatch")
    log("[OK] ✓ position_in_sentence equals bert_index")

    # 5) relative_position check per sentence
    log("[INFO] Verifying relative_position per sentence (tolerance {:.1e})".format(args.relative_tol))
    bad_rel = 0
    # cast bert_index as int for grouping
    df["_bert_index_int"] = converted["bert_index"].astype(int)
    for sid, g in df.groupby("sentence_id"):
        idxs = g.index
        max_idx = g["_bert_index_int"].max()
        if max_idx <= 0:
            # single-token sentence? accept relative_position = 0
            continue
        expected = g["_bert_index_int"] / float(max_idx)
        actual = rel_pos.loc[idxs].astype(float)
        # check absolute diff
        diff = (expected - actual).abs()
        if (diff > args.relative_tol).any():
            bad = idxs[diff > args.relative_tol].tolist()[:5]
            log(f"[ERROR] Relative position mismatch for sentence {sid}. Examples indices: {bad}")
            bad_rel += len(bad)
            # break early
            break
    if bad_rel:
        fatal(log, f"{bad_rel} relative_position mismatches found.")
    else:
        log("[OK] ✓ relative_position consistent with bert_index / max_index per sentence")

    # 6) num_subtokens consistency
    log("[INFO] Checking num_subtokens consistency by grouping (per sentence_id, word_id)")

    # valid rows for grouping: those with a word_id
    grouped = df[df["word_id"].apply(lambda x: not (pd.isna(x) or x==""))]
    # parse word_id int
    grouped = grouped.copy()
    grouped["word_id_parsed"] = grouped["word_id"].apply(parse_word_id)
    # drop rows where word_id parse failed
    grouped_valid = grouped[grouped["word_id_parsed"].notna()]

    if grouped_valid.empty:
        log("[WARN] No rows with valid word_id found. Skipping num_subtokens grouping check.")
    else:
        # compute counts
        grp_counts = grouped_valid.groupby(["sentence_id","word_id_parsed"]).size().rename("count").reset_index()
        # build a lookup
        lookup = {(int(r["sentence_id"]), int(r["word_id_parsed"])): int(r["count"]) for _, r in grp_counts.iterrows()}

        # iterate rows with a parsed word_id and compare
        inconsistencies = []
        big_values = []
        for idx, row in grouped_valid.iterrows():
            sid = int(row["sentence_id"])
            wid = int(row["word_id_parsed"])
            reported = int(float(row["num_subtokens"])) if not pd.isna(row["num_subtokens"]) else None
            expected = lookup.get((sid,wid), None)
            if expected is None:
                # shouldn't happen
                inconsistencies.append((idx, sid, wid, reported, expected))
                continue
            # reported for specials should be -1, but grouped_valid excludes specials
            if reported != expected:
                inconsistencies.append((idx, sid, wid, reported, expected))
            if expected > args.max_subtoken_warn:
                big_values.append((sid,wid,expected))

        if inconsistencies:
            sample = inconsistencies[:10]
            log(f"[ERROR] num_subtokens mismatch found for {len(inconsistencies)} rows. Examples:")
            for it in sample:
                idx,sid,wid,rep,exp = it
                log(f"  idx={idx} sentence={sid} word_id={wid} reported={rep} expected_group_count={exp}")
            fatal(log, "num_subtokens grouping inconsistent")
        else:
            log("[OK] ✓ num_subtokens matches counts computed per (sentence_id, word_id)")

        if big_values:
            log("[WARN] Found unusually large num_subtokens for some (sentence,word). Examples (sentence,word,count):")
            for sid,wid,c in big_values[:10]:
                log(f"  {sid},{wid},{c}")
            log(f"[WARN] threshold for warning = {args.max_subtoken_warn}")

    # 7) specials num_subtokens check: specials should have num_subtokens == -1 (by convention)
    specials_df = df[tokens.isin(["[CLS]","[SEP]","[PAD]","[UNK]"])]
    if not specials_df.empty:
        bad_specials = specials_df[specials_df["num_subtokens"].astype(float) != -1.0]
        if not bad_specials.empty:
            log(f"[ERROR] Found special tokens with num_subtokens != -1. Examples (first 10):")
            for i, r in bad_specials.head(10).iterrows():
                log(f"  idx={i} token={r['bert_token']} num_subtokens={r['num_subtokens']}")
            fatal(log, "Special tokens must have num_subtokens == -1 (convention used)")
    log("[OK] ✓ Special tokens have num_subtokens == -1 as expected")

    # 8) sanity bounds for num_subtokens overall
    numeric_subtokens = num_subtokens.dropna().astype(float)
    if (numeric_subtokens < -1).any():
        fatal(log, "num_subtokens contains values < -1 (invalid)")
    # warn for very large but not fatal
    above_reasonable = numeric_subtokens[numeric_subtokens > args.max_subtoken_warn]
    if len(above_reasonable):
        log(f"[WARN] {len(above_reasonable)} tokens have num_subtokens > {args.max_subtoken_warn}. See log for examples.")
    else:
        log("[OK] ✓ num_subtokens values within reasonable bounds")

    # 9) additional checks
    # - no empty token strings
    empty_tokens = tokens.str.strip() == ""
    if empty_tokens.any():
        idxs = df.index[empty_tokens][:10].tolist()
        log(f"[ERROR] Found empty bert_token strings at indices: {idxs}")
        fatal(log, "Empty bert_token strings found")
    log("[OK] ✓ No empty bert_token values")

    # - check that for tokens with same (sentence_id,word_id) all num_subtokens identical
    pairs = df[df["word_id"].apply(lambda x: not (pd.isna(x) or str(x).strip()==""))][["sentence_id","word_id","num_subtokens"]].copy()
    if not pairs.empty:
        pairs["word_id_parsed"] = pairs["word_id"].apply(parse_word_id)
        coll = pairs.groupby(["sentence_id","word_id_parsed"])["num_subtokens"].nunique()
        bad = coll[coll > 1]
        if not bad.empty:
            log("[ERROR] Inconsistent num_subtokens reported for the same (sentence_id,word_id). Examples:")
            for (sid,wid), nvals in bad.head(10).items():
                log(f"  sentence={sid} word_id={wid} distinct_num_subtokens={nvals}")
            fatal(log, "Inconsistent num_subtokens within same group")
    log("[OK] ✓ num_subtokens is consistent across tokens belonging to same word")

    # final summary
    log("\n[INFO] === VERIFICATION COMPLETE ===")
    log("[INFO] All checks passed (or warnings reported).")
    logfile_handle.close()
    print(f"[INFO] Detailed log written to: {args.log_file}")
    return 0

if __name__ == "__main__":
    main()
