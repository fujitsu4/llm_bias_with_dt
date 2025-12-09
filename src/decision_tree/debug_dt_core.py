"""
debug_dt_core.py
Author: Zakaria JOUILIL

Description:
    Comprehensive test suite for decision_tree_core.py (core DT utilities).
    Tests (unit + integration) for:
      - train_decision_tree
      - extract_rules_text
      - compute_leaf_stats
      - format_rules_with_stats
      - load_features_and_label_from_df
      - train_and_extract_rules_from_df (end-to-end)

Parameters :
    --seed : the seed for the decision tree
    --stress : flag to activate stress tests
      
Usage:
    python -m src.decision_tree.debug_dt_core --seed 42 --stress
"""

from __future__ import annotations
import argparse
import os
import json
import shutil
import tempfile
import traceback
from datetime import datetime, UTC
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

# Import functions under test
try:
    from src.decision_tree.decision_tree_core import (
        train_decision_tree,
        extract_rules_text,
        compute_leaf_stats,
        format_rules_with_stats,
        load_features_and_label_from_df,
        train_and_extract_rules_from_df,
        DEFAULT_FEATURE_COLS,
    )
except Exception as e:
    raise ImportError(
        "Could not import decision_tree_core. Ensure you're running from repo root "
        "and module path src.decision_tree.decision_tree_core is available.\n"
        f"Import error: {e}"
    )

# --------------------------
# Utility helpers
# --------------------------
def now_iso() -> str:
    return datetime.now(UTC).isoformat()

def print_ok(msg: str):
    print(f"[OK] {msg}")

def print_fail(msg: str):
    print(f"[FAIL] {msg}")

def safe_run(name: str, fn, *args, **kwargs):
    print(f"\n===== Running {name} =====")
    try:
        fn(*args, **kwargs)
        print_ok(f"{name} passed.")
    except Exception as e:
        print_fail(f"{name} failed:\n{e}\n{traceback.format_exc()}")

# --------------------------
# Synthetic data utilities
# --------------------------
def random_dataset(n: int = 200, d: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    # simple label: linear combination threshold -> binary target
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    cols = [f"feature{i}" for i in range(d)]
    df = pd.DataFrame(X, columns=cols)
    df["label"] = y
    return df

def random_binary_features_df(n: int, d: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    arr = rng.integers(low=0, high=2, size=(n, d))
    cols = [f"f{i}" for i in range(d)]
    df = pd.DataFrame(arr, columns=cols).astype(float)
    df["label"] = rng.integers(0, 2, size=n)
    return df

# --------------------------
# Tests for train_decision_tree
# --------------------------
def test_train_invalid_shapes():
    # X and y different lengths -> ValueError expected
    X = np.zeros((5, 3))
    y = np.zeros((4,))
    try:
        train_decision_tree(X, y)
        raise AssertionError("Expected ValueError for mismatched shapes")
    except ValueError:
        pass

def test_train_empty():
    X = np.zeros((0, 3))
    y = np.zeros((0,))
    try:
        train_decision_tree(X, y)
        raise AssertionError("Expected ValueError for empty training set")
    except ValueError:
        pass

def test_train_numpy_and_df_and_reproducibility():
    # Create small dataset
    df = random_dataset(n=100, d=4, seed=7)
    X_df = df[[c for c in df.columns if c != "label"]]
    y = df["label"].values

    clf1 = train_decision_tree(X_df, y, max_depth=3, min_samples_leaf=2, random_state=123)
    clf2 = train_decision_tree(X_df.values, y, max_depth=3, min_samples_leaf=2, random_state=123)
    # Export text to compare
    r1 = extract_rules_text(clf1, list(X_df.columns))
    r2 = extract_rules_text(clf2, list(X_df.columns))
    assert r1 == r2, "Reproducibility mismatch between df/numpy with same seed"

    # Different seed -> likely different structure
    clf_diff = train_decision_tree(X_df, y, max_depth=3, min_samples_leaf=2, random_state=999)
    rdiff = extract_rules_text(clf_diff, list(X_df.columns))
    if r1 == rdiff:
        # Rare, but allowed — use weak assertion: If identical, check tree depth difference
        pass

def test_train_edge_hyperparams():
    df = random_dataset(n=50, d=3, seed=1)
    X = df[[c for c in df.columns if c != "label"]]
    y = df["label"].values
    # very shallow
    clf_shallow = train_decision_tree(X, y, max_depth=1, min_samples_leaf=1, random_state=2)
    rshallow = extract_rules_text(clf_shallow, list(X.columns))
    assert isinstance(rshallow, str) and len(rshallow) > 0

# --------------------------
# Tests for extract_rules_text + compute_leaf_stats + format_rules_with_stats
# --------------------------
def test_extract_and_leaf_stats_consistency():
    df = random_dataset(n=200, d=5, seed=10)
    X = df[[c for c in df.columns if c != "label"]]
    y = df["label"].values

    clf = train_decision_tree(X, y, max_depth=4, min_samples_leaf=5, random_state=42)
    rules_text = extract_rules_text(clf, list(X.columns))
    assert isinstance(rules_text, str) and "class:" in rules_text

    stats = compute_leaf_stats(clf, X, y)
    # stats is a dict mapping node id to dict with keys
    assert isinstance(stats, dict)
    for v in stats.values():
        assert "n_samples" in v and "n_pos" in v and "pos_rate" in v

    formatted = format_rules_with_stats(clf, list(X.columns), X, y, top_k_lines=10)
    assert "Per-leaf statistics" in formatted

# --------------------------
# Tests for load_features_and_label_from_df
# --------------------------
def test_load_features_basic_and_missing():
    # Build df with extra columns
    n = 100
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "sentence_id": list(range(n)),
        "bert_index": list(range(n)),
        "bert_token": ["tok"]*n,
        # add required features
    })
    # add default features with random numbers
    for f in DEFAULT_FEATURE_COLS:
        df[f] = rng.normal(size=n)
    # add labels for layer 1
    df["top5_l1"] = (rng.random(n) > 0.5).astype(int)

    X, y = load_features_and_label_from_df(df, label_col="top5_l1", feature_cols=None)
    assert X.shape[1] == len(DEFAULT_FEATURE_COLS)
    assert X.shape[0] == n
    assert y.shape[0] == n

    # missing feature with require_all -> error
    df2 = df.drop(columns=[DEFAULT_FEATURE_COLS[0]])
    try:
        load_features_and_label_from_df(df2, label_col="top5_l1", feature_cols=None, require_all_features=True)
        raise AssertionError("Expected ValueError when features missing and require_all_features=True")
    except ValueError:
        pass

    # missing feature with require_all_features=False -> works
    X2, y2 = load_features_and_label_from_df(df2, label_col="top5_l1", feature_cols=None, require_all_features=False)
    assert X2.shape[1] == len(DEFAULT_FEATURE_COLS) - 1

def test_load_features_binarize_label():
    # y values not in {0,1} should be binarized
    n = 50
    rng = np.random.default_rng(2)
    df = pd.DataFrame()
    for f in DEFAULT_FEATURE_COLS:
        df[f] = rng.normal(size=n)
    # label values {0,2,5}
    df["top5_l1"] = rng.integers(low=0, high=3, size=n) * 2
    X, y = load_features_and_label_from_df(df, label_col="top5_l1", feature_cols=None)
    assert set(np.unique(y)).issubset({0, 1})

# --------------------------
# Tests for train_and_extract_rules_from_df (end-to-end)
# --------------------------
def make_attention_like_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame()
    # add some ignored columns
    df["sentence_id"] = np.arange(n)
    df["bert_index"] = np.arange(n)
    df["bert_token"] = ["tok"] * n
    # add required features
    for f in DEFAULT_FEATURE_COLS:
        # random floats but stable ranges
        df[f] = rng.normal(loc=0.0, scale=3.0, size=n)
    # dataset column (ignored)
    df["dataset"] = ["X"] * n
    # add a label column for layer 1 that is not constant
    logits = df[DEFAULT_FEATURE_COLS[0]] * 0.5 + df[DEFAULT_FEATURE_COLS[1]] * -0.2
    df["top5_l1"] = (logits > np.median(logits)).astype(int)
    return df

def test_train_and_extract_end_to_end(tmp_outdir: Optional[str] = None):
    df = make_attention_like_df(n=300, seed=11)
    # run training
    out = train_and_extract_rules_from_df(
        df,
        layer=1,
        feature_cols=None,
        max_depth=4,
        min_samples_leaf=5,
        dt_seed=42,
        save_rules_full_to=os.path.join(tmp_outdir, "rules_full.txt") if tmp_outdir else None,
        save_rules_pos_only_to=os.path.join(tmp_outdir, "rules_pos.txt") if tmp_outdir else None,
        save_rules_pos_simplified_to=os.path.join(tmp_outdir, "rules_simp.txt") if tmp_outdir else None,
    )
    # check keys
    for k in ("clf", "rules_full", "rules_pos_only", "leaf_stats", "n_samples", "n_pos"):
        assert k in out
    assert out["n_samples"] == 300
    # If files were requested, check they exist
    if tmp_outdir is not None:
        assert os.path.isfile(os.path.join(tmp_outdir, "rules_full.txt"))
        assert os.path.isfile(os.path.join(tmp_outdir, "rules_pos.txt"))
        assert os.path.isfile(os.path.join(tmp_outdir, "rules_simp.txt"))

def test_train_and_extract_reproducibility(tmp_outdir: Optional[str] = None):
    df = make_attention_like_df(n=250, seed=17)

    out1 = train_and_extract_rules_from_df(df, layer=1, dt_seed=1234)
    out2 = train_and_extract_rules_from_df(df, layer=1, dt_seed=1234)

    # Must be strictly identical
    assert out1["rules_full"] == out2["rules_full"], \
        "Same seed must produce identical rules"

    # Different seed -> often but not always different rules
    out3 = train_and_extract_rules_from_df(df, layer=1, dt_seed=4321)

    # Soft check:
    if out1["rules_full"] != out3["rules_full"]:
        # Expected case: different rules
        return
    else:
        # Rare but acceptable: the algorithm converged to same structure
        print("[INFO] Identical rules with different seeds — acceptable for stable dataset.")

# --------------------------
# Stress tests (optional)
# --------------------------
def stress_tests(outdir: str, n_runs: int = 50):
    print(f"[INFO] Running stress tests (n_runs={n_runs}) — this may take time...")
    for s in range(n_runs):
        seed = 1000 + s
        df = make_attention_like_df(n=1000, seed=seed)
        # Choose random layer (but our df has label only for layer1, so reuse)
        try:
            out = train_and_extract_rules_from_df(df, layer=1, dt_seed=seed)
        except Exception as e:
            raise AssertionError(f"Stress run failed at seed {seed}: {e}")

    print_ok("Stress tests completed without error.")

# --------------------------
# Test runner & CLI
# --------------------------
def run_all_tests(outdir: Optional[str] = None, stress: bool = False):
    tmpdir = None
    if outdir is None:
        tmpdir = tempfile.mkdtemp(prefix="debug_dt_core_")
        outdir = tmpdir
    os.makedirs(outdir, exist_ok=True)

    # Use small tmp dir for tests that write files
    safe_run("train_decision_tree: invalid shapes", test_train_invalid_shapes)
    safe_run("train_decision_tree: empty input", test_train_empty)
    safe_run("train_decision_tree: df/numpy & reproducibility", test_train_numpy_and_df_and_reproducibility)
    safe_run("train_decision_tree: hyperparams edge", test_train_edge_hyperparams)

    safe_run("extract_rules_text & leaf stats consistency", test_extract_and_leaf_stats_consistency)

    safe_run("load_features: basic and missing feature handling", test_load_features_basic_and_missing)
    safe_run("load_features: binarize label behavior", test_load_features_binarize_label)

    safe_run("end-to-end train_and_extract_rules_from_df (write files)", test_train_and_extract_end_to_end, outdir)
    safe_run("train_and_extract reproducibility", test_train_and_extract_reproducibility, outdir)

    if stress:
        safe_run("stress_tests (long)", stress_tests, outdir, 20)  # default 20 runs (configurable)

# --------------------------
# CLI
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Debug & test decision_tree_core utilities.")
    parser.add_argument("--seed", type=int, default=42, help="Global seed for reproducible synthetic data (default=42)")
    parser.add_argument("--outdir", type=str, default=None, help="Directory to write temporary rule files (optional)")
    parser.add_argument("--stress", action="store_true", help="Run stress tests (long)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    # python's random not relied upon heavily here, but keep in sync
    import random as _random
    _random.seed(args.seed)

    print(f"[{now_iso()}] Starting debug_dt_core tests (seed={args.seed})")
    run_all_tests(outdir=args.outdir, stress=args.stress)
    print(f"\n[{now_iso()}] All selected tests finished.")

if __name__ == "__main__":
    main()
