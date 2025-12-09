"""
verify_dt_reproductibility.py
Author : Zakaria JOUILIL

Description : 
    This script validates the reproductibility of the decision tree training and
    rule extraction pipeline. By : 
        - Checking that using the same random seed yields EXACTLY the same tree.
        - Checking that changing the random seed alters the structure of the tree.
        - Checking that rule extraction is stable under identical conditions.
        - Checking stability over multiple random datasets.
        - Checking that simplified rules remain stable under identical seeds.

Input :
    --input_csv : the real .csv file containg features and attention scores (compulsory for test 5)
    --seed : seed for the decision tree
Usage:
    python -m src.decision_tree.verify_dt_reproductibility \
    --input_csv /content/drive/MyDrive/results/attention_score/attention_top5_pretrained.csv \
    --seed 168652
"""

import numpy as np
import pandas as pd
import traceback
from src.decision_tree.decision_tree_core import (
    train_decision_tree,
    extract_rules_text,
    compute_leaf_stats,
    train_and_extract_rules_from_df,
    simplify_rules
)


# -------------------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------------------

def print_ok(msg):
    print(f"[OK] {msg}")

def print_fail(msg):
    print(f"[FAIL] {msg}")

def safe_run(test_name, fn):
    """Runs a test function and catches all exceptions."""
    print(f"\n===== Running {test_name} =====")
    try:
        fn()
        print_ok(f"{test_name} passed.")
    except Exception as e:
        print_fail(f"{test_name} failed:\n{e}\n{traceback.format_exc()}")


def random_dataset(n=200, d=5, seed=0):
    """Generate a random binary dataset with d features."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"feature{i}" for i in range(d)])
    df["label"] = y
    return df


# -------------------------------------------------------------------------
# TEST 1 — Identical seeds must yield identical rules
# -------------------------------------------------------------------------

def test_same_seed_same_tree():

    df = random_dataset(seed=42)

    rules1, stats1, clf1 = extract_with_seed(df, seed=42)
    rules2, stats2, clf2 = extract_with_seed(df, seed=42)

    assert rules1 == rules2, "Rules differ with identical seed."
    assert stats1 == stats2, "Leaf stats differ with identical seed."
    assert np.array_equal(clf1.tree_.feature, clf2.tree_.feature)
    assert np.allclose(clf1.tree_.threshold, clf2.tree_.threshold)
    
    simp1 = simplify_rules(rules1)
    simp2 = simplify_rules(rules2)
    assert simp1 == simp2

    print_ok("Rules and stats are identical with same seed.")


# -------------------------------------------------------------------------
# TEST 2 — Different seeds must yield different rules
# -------------------------------------------------------------------------

def test_different_seeds_different_tree():

    df = random_dataset(seed=42)

    rules1, stats1, clf1 = extract_with_seed(df, seed=0)
    rules2, stats2, clf2 = extract_with_seed(df, seed=99)

    assert rules1 != rules2, "Rules unexpectedly identical with different seeds."
    print_ok("Different seeds produce different tree structure.")


# -------------------------------------------------------------------------
# TEST 3 — Repeated identical calls must produce stable output
# -------------------------------------------------------------------------

def test_repeat_stability():

    df = random_dataset(seed=123)

    outputs = []
    for i in range(5):
        rules, stats, clf = extract_with_seed(df, seed=777)
        outputs.append(rules)

    for i in range(4):
        assert outputs[i] == outputs[i+1], "Instability detected on repeated calls."

    print_ok("Repeated training with same seed is stable.")


# -------------------------------------------------------------------------
# TEST 4 — Large number of random datasets check
# -------------------------------------------------------------------------

def test_multiple_random_datasets():

    for k in range(10):  # 10 datasets
        df = random_dataset(seed=k)
        rules1, _, _ = extract_with_seed(df, seed=1)
        rules2, _, _ = extract_with_seed(df, seed=1)
        assert rules1 == rules2, f"Inconsistent output on dataset #{k} with same seed."

    print_ok("All 10 random datasets produced stable outputs with same seed.")


# -------------------------------------------------------------------------
# TEST 5 — Full pipeline reproductibility test
# -------------------------------------------------------------------------
def test_full_pipeline(input_csv):

    # 1) Load the REAL dataset safely
    df = pd.read_csv(
        input_csv,
        sep=";",
        keep_default_na=False,
        na_values=["", " ", "NaN", "nan"],
        low_memory=False
    )

    # 2) Feature columns expected by the DT pipeline
    feature_cols = [
        "is_numeric", "is_punctuation", "is_stopword", "is_pronoun",
        "pos_id", "dep_id", "ent_id", "depth_in_tree", "num_dependents",
        "is_special_token", "token_length", "is_subword",
        "position_in_sentence", "relative_position", "num_subtokens",
        "token_relative_frequency", "word_sentence_frequency", "token_rank",
        "word_burstiness", "prev_pos_id", "next_pos_id"
    ]

    # 3) Convert feature columns to numeric
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # 4) Replace missing values by -1.0 (consistent with your dataset)
    df[feature_cols] = df[feature_cols].fillna(-1.0)

    # 5) Run pipeline twice with SAME seed
    out1 = train_and_extract_rules_from_df(df, layer=1, dt_seed=10)
    out2 = train_and_extract_rules_from_df(df, layer=1, dt_seed=10)

    # 6) Compare reproductibility
    if out1["rules_full"] != out2["rules_full"] or out1["leaf_stats"] != out2["leaf_stats"]:
        print("[FAIL] Test 5 - Full pipeline reproductibility failed:")
        raise AssertionError("Pipeline is not reproducible with same seed!")

# -------------------------------------------------------------------------
# Helper function for extraction
# -------------------------------------------------------------------------

def extract_with_seed(df, seed):
    """Train + extract rules with a given seed, return (rules_text, stats_dict, clf)."""
    X = df[[c for c in df.columns if c != "label"]]
    y = df["label"].values
    clf = train_decision_tree(X.values, y, random_state=seed)

    rules = extract_rules_text(clf, X.columns.tolist())
    stats = compute_leaf_stats(clf, X.values, y)
    return rules, stats, clf


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

def main():
    import argparse, random
    from datetime import datetime, UTC

    parser = argparse.ArgumentParser(description="Train a DT on a single layer top5 label (core debug).")
    parser.add_argument("--input_csv", required=True, help="CSV containing bert tokens + features + top5_l* columns")
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed to ensure reproducibility (default=42)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    start = datetime.now(UTC).isoformat()
    print(f"[{start}] Verifying Decision Tree rules reproductibility.")
    print(f"Using seed = {args.seed}")

    safe_run("Test 1 - Same seed → identical tree", test_same_seed_same_tree)
    safe_run("Test 2 - Different seed → different tree", test_different_seeds_different_tree)
    safe_run("Test 3 - Stability on repeated calls", test_repeat_stability)
    safe_run("Test 4 - Stability across 10 datasets", test_multiple_random_datasets)
    safe_run("Test 5 - Full pipeline reproducibility",lambda: test_full_pipeline(args.input_csv))

    print("\n--------------------------------")
    print(" reproductibility tests completed ")
    print("--------------------------------")


if __name__ == "__main__":
    main()
