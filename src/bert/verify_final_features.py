"""
verify_final_features.py
Author: Zakaria JOUILIL

Description:
    Exhaustive verification script for the final feature file (bert_final_features.csv)
    that will be used as input for the Decision Tree.

    It checks:
      - Exact column order & presence
      - Types and ranges for each column
      - Per-sentence invariants (relative positions sum, no missing word_ids, etc.)
      - Cross-feature consistency (special tokens conventions, num_subtokens grouping,
        propagation of word-level stats to sub-tokens, prev/next neighbors correctness)
      - Statistical sanity (word_burstiness in [-1,1], token_rank positive for non-specials)

Input :
    outputs/bert/bert_final_features.csv

Output : 
    A log file containing summary and report.

Usage:
    python -m src.bert.verify_final_features \
        --input outputs/bert/bert_final_features.csv \
        --log logs/bert_final_logs.txt
"""

import argparse
import sys
import os
import math
from datetime import datetime, timezone
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

# -----------------------
# Logger helper
# -----------------------
def setup_logger(log_file_path):
    """Redirect stdout/stderr to both console and a log file."""
    os.makedirs(os.path.dirname(log_file_path) or ".", exist_ok=True)
    log_f = open(log_file_path, "w", encoding="utf-8")

    class Logger:
        def __init__(self, terminal, logfile):
            self.terminal = terminal
            self.log = logfile

        def write(self, msg):
            self.terminal.write(msg)
            self.log.write(msg)
            self.log.flush()

        def flush(self):
            try:
                self.terminal.flush()
            except:
                pass
            try:
                self.log.flush()
            except:
                pass

    sys.stdout = Logger(sys.stdout, log_f)
    sys.stderr = sys.stdout

def ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

def info(msg):
    print(f"[{ts()}] [INFO] {msg}")

def ok(msg):
    print(f"[{ts()}] [OK] {msg}")

def warn(msg):
    print(f"[{ts()}] [WARN] {msg}")

def error(msg):
    print(f"[{ts()}] [ERROR] {msg}")

def fatal(msg, code=1):
    error(msg)
    error("Verification FAILED (fatal). Exiting.")
    sys.exit(code)

# -----------------------
# Main verification
# -----------------------
def verify(input_csv, log_file, relative_tol=1e-6, sample_sentences=200):
    setup_logger(log_file)
    info(f"Logging -> {log_file}")
    info("=== VERIFYING FINAL FEATURES ===")
    info(f"Reading CSV: {input_csv}")

    try:
        df = pd.read_csv(input_csv, sep=";", keep_default_na=False, na_values=[""])
    except Exception as e:
        fatal(f"Cannot read input CSV '{input_csv}': {e}")

    # expected exact order and names
    expected_cols = [
        "sentence_id","bert_index","bert_token","word_id",
        "is_numeric","is_punctuation","is_stopword","is_pronoun",
        "pos_id","dep_id","ent_id","depth_in_tree","num_dependents",
        "is_special_token","token_length","is_subword","position_in_sentence",
        "relative_position","num_subtokens","token_relative_frequency",
        "word_sentence_frequency","token_rank","word_burstiness",
        "prev_pos_id","next_pos_id","dataset"
    ]

    # 1) Columns: presence and order
    info("1) Checking columns presence and exact order")
    got_cols = list(df.columns)
    if got_cols != expected_cols:
        error("Column mismatch or wrong order.")
        error(f"Expected (len={len(expected_cols)}): {expected_cols}")
        error(f"Found    (len={len(got_cols)}): {got_cols}")
        # show minimal diff
        missing = [c for c in expected_cols if c not in got_cols]
        extra = [c for c in got_cols if c not in expected_cols]
        if missing:
            error(f"Missing columns: {missing}")
        if extra:
            error(f"Unexpected extra columns: {extra}")
        fatal("Columns are not exactly as expected. Reorder/rename the file before proceeding.")
    ok("Column names and order are EXACTLY correct.")

    n = len(df)
    info(f"Total rows: {n}")

    # Quick types conversions/casts with safe coercion (do not overwrite original unless safe)
    info("2) Basic type normalization & null checks")

    # numeric integer-like fields
    int_cols = ["sentence_id","bert_index","position_in_sentence"]
    for c in int_cols:
        try:
            df[c] = pd.to_numeric(df[c], errors="raise").astype(int)
        except Exception as e:
            fatal(f"Column {c} must be integer-like. Error: {e}")

    # bert_token strings
    df["bert_token"] = df["bert_token"].astype(str)

    # dataset string
    df["dataset"] = df["dataset"].astype(str)

    # boolean-like 0/1 fields
    bin_cols = ["is_numeric","is_punctuation","is_stopword","is_pronoun","is_special_token","is_subword"]
    for col in bin_cols:
        # 1) normalize strings for common boolean words
        s = df[col].astype(str).str.strip().replace({"True":"1","true":"1","False":"0","false":"0"})
        # 2) attempt numeric conversion (coerce invalid -> NaN)
        nums = pd.to_numeric(s, errors="coerce")
        # 3) if any NaNs remain, report examples and fail
        if nums.isna().any():
            sample_bad = s[nums.isna()].unique()[:10].tolist()
            fatal(f"Column {col} contains values that cannot be parsed as 0/1. Examples: {sample_bad}")
        # 4) round (safety) and cast to integer 0/1
        nums = nums.round().astype(int)
        # 5) final sanity: only 0 or 1 allowed
        unique_vals = set(nums.unique().tolist())
        if not unique_vals.issubset({0, 1}):
            fatal(f"Column {col} contains values outside {{0,1}} after normalization. Found: {unique_vals}")
        # 6) write back
        df[col] = nums

    ok("Boolean-like columns are 0/1 and normalized.")

    # floats
    float_cols = ["relative_position","token_relative_frequency","word_burstiness"]
    for c in float_cols:
        try:
            df[c] = pd.to_numeric(df[c], errors="raise").astype(float)
        except Exception as e:
            fatal(f"Column {c} must be numeric (float). Error: {e}")

    # ints (but possibly -1 placeholders)
    int_like_cols = ["word_id","pos_id","dep_id","ent_id","depth_in_tree","num_dependents","num_subtokens","token_rank","word_sentence_frequency"]
    for c in int_like_cols:
        # keep as numeric but allow NA for word_id
        if c == "word_id":
            # allow empty -> NaN
            df[c] = df[c].replace("", np.nan)
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            except Exception as e:
                fatal(f"Column {c} must be numeric-like (or empty for specials). Error: {e}")
        else:
            try:
                df[c] = pd.to_numeric(df[c], errors="raise").astype(int)
            except Exception as e:
                fatal(f"Column {c} must be integer-like. Error: {e}")

    ok("Numeric columns converted/validated.")

    # 3) Primary invariants
    info("3) Checking primary invariants and quick consistency tests")

    # a) bert_index equals position_in_sentence
    mism = (df["bert_index"].astype(int) != df["position_in_sentence"].astype(int))
    if mism.any():
        sample = df.loc[mism].head(10)[["sentence_id","bert_index","position_in_sentence","bert_token"]]
        error("Mismatch: bert_index != position_in_sentence for some rows. Examples:")
        print(sample.to_string(index=False))
        fatal("bert_index and position_in_sentence must be identical for all rows.")
    ok("bert_index == position_in_sentence for all rows.")

    # b) token_relative_frequency in [0,1]
    if df["token_relative_frequency"].lt(0).any() or df["token_relative_frequency"].gt(1).any():
        fatal("token_relative_frequency must be within [0,1].")
    ok("token_relative_frequency values are within [0,1].")

    # c) word_burstiness finite and in [-1,1]
    if not np.isfinite(df["word_burstiness"]).all():
        fatal("word_burstiness contains non-finite values.")
    if (df["word_burstiness"] < -1.0 - 1e-9).any() or (df["word_burstiness"] > 1.0 + 1e-9).any():
        fatal("word_burstiness must be within [-1,1].")
    ok("word_burstiness values are finite and within [-1,1].")

    # d) token_rank: non-special tokens must have rank >= 1; specials must have rank 0
    non_special_ranks = df.loc[df["is_special_token"] == 0, "token_rank"]
    if (non_special_ranks.fillna(-1).astype(int) < 1).any():
        bad = df.loc[(df["is_special_token"]==0) & (df["token_rank"]<1)].head(10)
        error("Non-special tokens must have token_rank >= 1. Examples:")
        print(bad[["sentence_id","bert_token","token_rank"]].to_string(index=False))
        fatal("token_rank invalid for some non-special tokens.")
    # specials check
    specials_with_nonzero_rank = df.loc[(df["is_special_token"]==1) & (df["token_rank"] != 0)]
    if not specials_with_nonzero_rank.empty:
        warn_count = len(specials_with_nonzero_rank)
        warn(f"{warn_count} special tokens have token_rank != 0 (expected convention: 0). Showing up to 10 examples:")
        print(specials_with_nonzero_rank.head(10)[["sentence_id","bert_token","token_rank"]].to_string(index=False))
        warn("This is a WARNING (not fatal). Check if that was intended.")

    ok("token_rank convention checked (non-special >=1).")

    # e) Specials: prev_pos_id and next_pos_id must equal -1 (by convention)
    specials = df[df["is_special_token"] == 1]
    if not specials.empty:
        bad_prev = specials.loc[specials["prev_pos_id"] != -1]
        bad_next = specials.loc[specials["next_pos_id"] != -1]
        if not bad_prev.empty or not bad_next.empty:
            error("Special tokens must have prev_pos_id and next_pos_id == -1 (convention). Examples:")
            print(pd.concat([bad_prev.head(5), bad_next.head(5)]).drop_duplicates().head(10).to_string(index=False))
            fatal("Special neighbors convention violated.")
    ok("Special tokens have prev_pos_id and next_pos_id == -1.")

    # f) For non-special tokens, pos_id/dep_id/ent_id should be integers >= 0 (or -1 if intentionally marked)
    non_specials = df[df["is_special_token"] == 0]
    for col in ["pos_id","dep_id","ent_id","depth_in_tree","num_dependents","num_subtokens","word_sentence_frequency"]:
        # allow -1 for some fields if that is used as placeholder, but ensure ints
        if non_specials[col].isna().any():
            error(f"Non-special tokens: column {col} contains NA values (not allowed).")
            fatal("Missing values in non-special token fields.")
    ok("Non-special token fields pos_id/dep_id/ent_id/depth_in_tree/num_dependents/num_subtokens/word_sentence_frequency are present (no NA).")

    # g) word_id: specials may have NaN, non-specials must have integer word_id >= 0
    ns = df[df["is_special_token"] == 0]
    if ns["word_id"].isna().any():
        bad = ns[ns["word_id"].isna()].head(10)
        error("Some non-special tokens have missing word_id. Examples:")
        print(bad[["sentence_id","bert_index","bert_token"]].to_string(index=False))
        fatal("word_id missing for non-special tokens.")
    # ensure integer-like and >=0
    if (ns["word_id"] < 0).any():
        fatal("Non-special tokens have negative word_id (invalid).")
    ok("word_id present and non-negative for all non-special tokens.")

    # 4) Per-sentence checks (sampled and exhaustive small checks)
    info("4) Per-sentence checks (sampled + critical full checks)")

    sent_ids = sorted(df["sentence_id"].unique())
    info(f"Unique sentences: {len(sent_ids)}")

    #a) Checking consistency of token_relative_frequency
    for sid, group in df.groupby("sentence_id"):
        non_special = group[group["is_special_token"] == 0]
        if len(non_special) == 0:
            continue

        N = len(non_special)

        # expected freq = count(token) / N
        counts = non_special["bert_token"].value_counts()

        for tok, expected_freq in (counts / N).items():
            actual = non_special.loc[non_special["bert_token"] == tok, "token_relative_frequency"]
            if not np.allclose(actual, expected_freq, atol=1e-6):
                fatal(f"token_relative_frequency mismatch in sentence {sid} for token '{tok}': "
                  f"expected {expected_freq}, got {actual.unique()}")

    # b) Check that for each (sentence_id, word_id), num_subtokens equals number of tokens mapped to that word
    info("4.b) Checking num_subtokens grouping consistency (exhaustive).")
    # Build grouping for non-special tokens
    grouped = df[df["is_special_token"] == 0].groupby(["sentence_id","word_id"])
    inconsistent = []
    for (sid,wid), g in grouped:
        reported_vals = pd.unique(g["num_subtokens"].astype(int))
        if len(reported_vals) != 1:
            inconsistent.append((sid,wid,reported_vals.tolist(), len(g)))
            if len(inconsistent) > 10:
                break
        else:
            reported = int(reported_vals[0])
            expected = len(g)
            if reported != expected:
                inconsistent.append((sid,wid,reported,expected))
                if len(inconsistent) > 10:
                    break
    if inconsistent:
        error(f"Found inconsistent num_subtokens for some (sentence_id,word_id). Showing up to 10 examples:")
        for ex in inconsistent[:10]:
            print("  ", ex)
        fatal("num_subtokens inconsistency detected.")
    ok("num_subtokens is consistent and equals group size for each (sentence_id,word_id).")

    # c) word_sentence_frequency propagation: ensure tokens belonging to same (sid,wid) share same word_sentence_frequency
    info("4.c) Checking word_sentence_frequency propagation to sub-tokens.")
    grouped_wsf = df[df["is_special_token"]==0].groupby(["sentence_id","word_id"])["word_sentence_frequency"].nunique()
    bad = grouped_wsf[grouped_wsf > 1]
    if not bad.empty:
        error("word_sentence_frequency not consistently propagated to all subtokens of a word. Examples:")
        print(bad.head(10).to_string())
        fatal("word_sentence_frequency propagation inconsistency.")
    ok("word_sentence_frequency is consistently propagated to sub-tokens.")

    # d) prev_pos_id / next_pos_id correctness for non-special tokens:
    info("4.d) Checking prev_pos_id / next_pos_id correctness for non-special tokens (exhaustive sample).")
    # Build per-sentence mapping from word_id -> pos_id
    neighbor_errors = []
    for sid in tqdm(sent_ids[:sample_sentences], desc="checking neighbors (sample)"):
        sub = df[df["sentence_id"] == sid]
        # only non-special tokens define word positions
        word_map = {}
        # for each unique word_id get pos_id (they should be same across subtokens)
        nonsp = sub[sub["is_special_token"] == 0]
        if nonsp.empty:
            continue
        # extract mapping: word_id -> pos_id (should be unique)
        grouped_word = nonsp.groupby("word_id")["pos_id"].nunique()
        if (grouped_word > 1).any():
            neighbor_errors.append((sid, "pos_id inconsistent across subtokens"))
            continue
        # get one representative pos_id per word_id
        rep = nonsp.groupby("word_id")["pos_id"].first().to_dict()
        # check every non-special token's prev/next against rep
        for _, row in nonsp.iterrows():
            wid = int(row["word_id"])
            expected_prev = rep.get(wid-1, -1)
            expected_next = rep.get(wid+1, -1)
            if int(row["prev_pos_id"]) != int(expected_prev):
                neighbor_errors.append((sid, row["bert_index"], "prev", wid, int(row["prev_pos_id"]), int(expected_prev)))
                if len(neighbor_errors) > 10:
                    break
            if int(row["next_pos_id"]) != int(expected_next):
                neighbor_errors.append((sid, row["bert_index"], "next", wid, int(row["next_pos_id"]), int(expected_next)))
                if len(neighbor_errors) > 10:
                    break
        if len(neighbor_errors) > 0:
            break

    if neighbor_errors:
        error("Neighbor (prev/next) pos_id mismatches detected (sample). Examples:")
        for ex in neighbor_errors[:10]:
            print("  ", ex)
        fatal("prev_pos_id / next_pos_id verification failed on sample. Investigate mapping logic.")
    ok("prev_pos_id and next_pos_id appear consistent with word-level pos_id on sampled sentences.")

    # e) Ensure no spaCy words were lost in mapping: For each sentence, the set of unique word_id for non-special tokens
    #    should form a contiguous range 0..max_word_id (this is expected from our earlier pipeline - warn if not).
    info("4.e) Checking contiguous word_id ranges per sentence (warn on gaps).")
    gaps = []
    for sid in sent_ids[:sample_sentences]:
        sub = df[(df["sentence_id"]==sid) & (df["is_special_token"]==0)]
        if sub.empty:
            continue
        wids = sorted(sub["word_id"].dropna().astype(int).unique().tolist())
        if not wids:
            continue
        if wids[0] != 0 or wids[-1] != max(wids):
            # maybe word indexing starts not at zero per original dataset; we only warn
            gaps.append((sid, wids[:5], wids[-5:]))
        else:
            # check full contiguous
            exp = list(range(wids[0], wids[-1]+1))
            if wids != exp:
                gaps.append((sid, wids[:5], wids[-5:]))
        if len(gaps) >= 5:
            break
    if gaps:
        warn("Found sentences with non-contiguous or non-zero-starting word_id ranges (sample). Examples:")
        for ex in gaps[:5]:
            print("  ", ex)
        warn("This is a WARNING (not fatal). If your pipeline expects contiguous 0..N indexing, investigate.")
    else:
        ok("word_id ranges appear contiguous and start at 0 on sampled sentences.")

    # 5) Cross-feature logical checks
    info("5) Cross-feature logical checks (a set of logical rules)")

    # Rule A: If is_pronoun==1 then pos_id should correspond to PRON tag id (we cannot map string -> id here,
    # but we can ensure is_pronoun implies pos_id is present and non-negative)
    pronoun_viol = df[(df["is_pronoun"]==1) & (df["pos_id"] < 0)]
    if not pronoun_viol.empty:
        error("Some tokens marked as is_pronoun==1 have invalid pos_id (<0). Examples:")
        print(pronoun_viol.head(10)[["sentence_id","bert_index","bert_token","is_pronoun","pos_id"]].to_string(index=False))
        fatal("is_pronoun implies invalid pos_id for some entries.")
    ok("is_pronoun implies valid pos_id where applicable.")

    # Rule B: punctuation flags vs token content
    punc_viol = df[(df["is_punctuation"]==1) & (~df["bert_token"].str.match(r'^\W+$'))]
    if len(punc_viol) > 0:
        warn_cnt = min(10, len(punc_viol))
        warn(f"{len(punc_viol)} rows flagged is_punctuation==1 but token text isn't pure punctuation. Showing up to {warn_cnt} examples:")
        print(punc_viol.head(warn_cnt)[["sentence_id","bert_index","bert_token"]].to_string(index=False))
        # Not fatal (depends on tokenization rules), but warn.
    else:
        ok("is_punctuation flags consistent with token strings (quick check).")

    # Rule C: is_numeric matches token characters (quick heuristic)
    num_viol = df[(df["is_numeric"]==1) & (~df["bert_token"].str.replace(r'[^\d\.]','', regex=True).str.len().gt(0))]
    if len(num_viol) > 0:
        warn(f"{len(num_viol)} tokens flagged is_numeric==1 but token doesn't look numeric. Showing up to 10 examples:")
        print(num_viol.head(10)[["sentence_id","bert_index","bert_token"]].to_string(index=False))
    else:
        ok("is_numeric flags consistent with token strings (quick heuristic).")

    # 6) Sanity summaries and final verdict
    info("6) Summary diagnostics (final)")

    unique_sent = df["sentence_id"].nunique()
    unique_tokens = df["bert_token"].nunique()
    n_specials = int(df["is_special_token"].sum())
    info(f"Unique sentences: {unique_sent}")
    info(f"Unique bert_token types: {unique_tokens}")
    info(f"Total rows: {n}")
    info(f"Special tokens (rows): {n_specials}")

    ok("Basic summary computed.")

    # Done
    info("=== ALL CHECKS COMPLETED ===")
    info("If no [ERROR]/fatal occurred above, file is verified (warnings may be present).")
    return 0

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify final merged features (bert_final_features.csv)")
    parser.add_argument("--input", required=True, help="Input CSV (final merged features)")
    parser.add_argument("--log_file", required=True, help="Path to save verbose verification log")
    parser.add_argument("--relative_tol", type=float, default=1e-6, help="Tolerance when checking sums like relative frequencies")
    parser.add_argument("--sample_sentences", type=int, default=200, help="Number of sentences to sample for some expensive checks")
    args = parser.parse_args()

    verify(args.input, args.log_file, relative_tol=args.relative_tol, sample_sentences=args.sample_sentences)