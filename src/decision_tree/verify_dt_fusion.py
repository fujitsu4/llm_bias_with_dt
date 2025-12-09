"""
verify_dt_fusion.py
Author: Zakaria JOUILIL

Description :
    Complete test script for verifying rule merging and simplification.
    It includes : generation of synthetic rule trees, assertions and expected-result
    logic and printing of first example per test group.
    The six test suites are :
        * Test 1 - No complementarity (no merge)
        * Test 2 - Simple complementarity
        * Test 3 - Deep complementarity
        * Test 4 - Multi-level complementarity
        * Test 5 - Floating precision
        * Test 6 - Indentation-insensitive

Usage :
    python -m src.decision_tree.verify_dt_fusion --seed 100
"""

import argparse
import random
import numpy as np
from typing import List, Dict, Any
from src.decision_tree.decision_tree_core import simplify_rules
from datetime import datetime, UTC

# -------------------------
# Helper: convert dict-rules -> sklearn-like text blocks
# -------------------------
def dict_rules_to_text(rules: list) -> str:
    """
    Convert list-of-dicts rules into a single text string formatted like sklearn export_text.
    Each rule is a dict: {"conditions":[{"feature":..., "op":..., "threshold":...}, ...], "class": 0|1}
    Returns blocks separated by a blank line.
    """
    blocks = []
    for rule in rules:
        lines = []
        for depth, cond in enumerate(rule.get("conditions", [])):
            indent = "|   " * depth
            # format threshold with a stable representation (6 decimals)
            thr = cond.get("threshold")
            if isinstance(thr, float):
                thr_s = f"{thr:.6f}".rstrip("0").rstrip(".") if thr is not None else "None"
            else:
                thr_s = str(thr)
            lines.append(f"{indent}|--- {cond['feature']} {cond['op']} {thr_s}")
        # leaf line
        indent = "|   " * len(rule.get("conditions", []))
        lines.append(f"{indent}|--- class: {int(rule.get('class',0))}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# -------------------------
# Helper: normalize text -> list of blocks (for canonical comparison)
# -------------------------
import re
def text_to_blocks(text: str) -> list:
    """
    Convert exported/simplified text into a canonical list of blocks.
    Each block is a tuple of normalized lines.
    Normalization rules:
      - trim spaces, collapse multiple spaces to one
      - normalize numeric formatting to float with 6 decimals
      - preserve operator and feature name
    Returns list of blocks (each block = tuple(lines...))
    """
    def norm_line(ln: str) -> str:
        # Remove leading/trailing whitespace
        s = ln.strip()
        # collapse multiple spaces
        s = re.sub(r"\s+", " ", s)
        # normalize numbers: find numbers and format
        def fmt_num(m):
            try:
                v = float(m.group(0))
                # format with 6 decimals then strip trailing zeros
                snum = f"{v:.6f}".rstrip("0").rstrip(".")
                return snum
            except:
                return m.group(0)
        s = re.sub(r"-?\d+\.\d+|-?\d+", fmt_num, s)
        return s

    raw_blocks = [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    out = []
    for b in raw_blocks:
        lines = [norm_line(l) for l in b.splitlines() if l.strip()]
        out.append(tuple(lines))
    return out

# ---------------------------------------------------------------------
# Utility: generate a random threshold
# ---------------------------------------------------------------------
def rand_threshold(low=-10, high=10):
    return round(random.uniform(low, high), 6)

# ---------------------------------------------------------------------
# Utility: create a path (list of conditions) ending in class 1 or 0
# Each condition is a dict: {"feature": str, "op": str, "threshold": float}
# ---------------------------------------------------------------------
def build_path(depth: int, force_feature: str = None, force_equal_prefix=None, complementary=False):
    path = []
    for i in range(depth):
        if force_equal_prefix and i < len(force_equal_prefix):
            # reuse forced prefix
            path.append(force_equal_prefix[i])
        else:
            feat = force_feature if force_feature else f"feature_{i+1}"
            op = random.choice(["<=", ">"])
            thr = rand_threshold()
            path.append({"feature": feat, "op": op, "threshold": thr})

    # if complementary at last node
    if complementary:
        last = path[-1]
        if last["op"] == "<=":
            last["op"] = ">"
        else:
            last["op"] = "<="

    return path

# ---------------------------------------------------------------------
# Convert a path to rule-format used by simplify_rules
# ---------------------------------------------------------------------
def path_to_rule(path: List[Dict[str, Any]], cls=1):
    return {
        "conditions": path,
        "class": cls
    }

# ---------------------------------------------------------------------
# Test 1 – No complementarity → no merge
# ---------------------------------------------------------------------
def generate_test1(n=10):
    tests = []
    for _ in range(n):
        # two unrelated features at top
        path1 = build_path(2, force_feature="A1")
        path2 = build_path(2, force_feature="B1")
        r1 = path_to_rule(path1, cls=1)
        r2 = path_to_rule(path2, cls=1)
        tests.append([r1, r2])
    return tests

# ---------------------------------------------------------------------
# Test 2 – Simple complementarity at depth=2
# ---------------------------------------------------------------------
def generate_test2(n=10):
    tests = []
    for _ in range(n):
        prefix = [
            {"feature": "A", "op": random.choice(["<=", ">"]), "threshold": rand_threshold()}
        ]
        node1 = {"feature": "B", "op": "<=", "threshold": rand_threshold()}
        node2 = {"feature": "B", "op": ">", "threshold": node1["threshold"]}
        r1 = path_to_rule(prefix + [node1], cls=1)
        r2 = path_to_rule(prefix + [node2], cls=1)
        tests.append([r1, r2])
    return tests

# ---------------------------------------------------------------------
# Test 3 – Deep complementarity (depth 3 or 4)
# ---------------------------------------------------------------------
def generate_test3(n=10):
    tests = []
    for _ in range(n):
        depth = random.choice([3, 4])
        prefix = []
        # generate prefix for depth-1
        for i in range(depth - 1):
            prefix.append({
                "feature": f"F{i}",
                "op": random.choice(["<=", ">"]),
                "threshold": rand_threshold()
            })
        # last complementary
        node1 = {"feature": f"F{depth}", "op": "<=", "threshold": rand_threshold()}
        node2 = {"feature": f"F{depth}", "op": ">", "threshold": node1["threshold"]}

        r1 = path_to_rule(prefix + [node1], cls=1)
        r2 = path_to_rule(prefix + [node2], cls=1)
        tests.append([r1, r2])
    return tests

# ---------------------------------------------------------------------
# Test 4 – Multi-level complementarity (needs two passes)
# ---------------------------------------------------------------------
def generate_test4(n=10):
    tests = []
    for _ in range(n):
        # C-branch
        tc = rand_threshold()
        c1 = [
            {"feature": "A", "op": "<=", "threshold": 10},
            {"feature": "B", "op": "<=", "threshold": 5},
            {"feature": "C", "op": "<=", "threshold": tc},
        ]
        c2 = [
            {"feature": "A", "op": "<=", "threshold": 10},
            {"feature": "B", "op": "<=", "threshold": 5},
            {"feature": "C", "op": ">", "threshold": tc},
        ]
        # D-branch
        td = rand_threshold()
        d1 = [
            {"feature": "A", "op": "<=", "threshold": 10},
            {"feature": "B", "op": ">", "threshold": 5},
            {"feature": "D", "op": "<=", "threshold": td},
        ]
        d2 = [
            {"feature": "A", "op": "<=", "threshold": 10},
            {"feature": "B", "op": ">", "threshold": 5},
            {"feature": "D", "op": ">", "threshold": td},
        ]
        tests.append([
            path_to_rule(c1, 1), path_to_rule(c2, 1),
            path_to_rule(d1, 1), path_to_rule(d2, 1)
        ])
    return tests

# ---------------------------------------------------------------------
# Test 5 – Floating precision
# ---------------------------------------------------------------------
def generate_test5(n=10):
    tests = []
    for _ in range(n):
        thresh = rand_threshold()
        small = thresh
        close = thresh - 1e-6
        # merge case
        r1 = path_to_rule([{ "feature": "C", "op": "<=", "threshold": small }], 1)
        r2 = path_to_rule([{ "feature": "C", "op": ">", "threshold": close }], 1)
        # non-merge case
        far = thresh + 0.1
        r3 = path_to_rule([{ "feature": "C", "op": "<=", "threshold": small }], 1)
        r4 = path_to_rule([{ "feature": "C", "op": ">", "threshold": far }], 1)
        tests.append([r1, r2, r3, r4])
    return tests

# ---------------------------------------------------------------------
# Test 6 – Indentation differences (ignored)
# Here represented simply as rules; indentation not relevant in structured format.
# ---------------------------------------------------------------------
def generate_test6(n=10):
    tests = []
    for _ in range(n):
        t = rand_threshold()
        r1 = path_to_rule([
            {"feature": "A", "op": "<=", "threshold": t},
            {"feature": "B", "op": "<=", "threshold": t}
        ], 1)
        r2 = path_to_rule([
            {"feature": "A", "op": "<=", "threshold": t},
            {"feature": "B", "op": ">", "threshold": t}
        ], 1)
        tests.append([r1, r2])
    return tests

# ---------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------
def run_test_case(name: str, generator, expected_logic):
    print(f"\n===== Running {name} =====")
    test_sets = generator()
    #print("Example input:", test_sets[0])
    print("Example input (text form):\n", dict_rules_to_text(test_sets[0]), "\n")


    for i, rules in enumerate(test_sets):
        # Convert to text
        rules_text = dict_rules_to_text(rules)

        # Apply simplifier
        simplified_text = simplify_rules(rules_text)

        # Normalize into comparable blocks
        orig_blocks = text_to_blocks(rules_text)
        simp_blocks = text_to_blocks(simplified_text)

        if not expected_logic(orig_blocks, simp_blocks):
            print(f"[FAIL] {name} – Case {i}")
            print("Original text:\n", rules_text)
            print("Simplified text:\n", simplified_text)
            print("Original blocks:", orig_blocks)
            print("Simplified blocks:", simp_blocks)
            return

    print(f"[OK] {name}")

# ---------------------------------------------------------------------
# Expected logic per test
# ---------------------------------------------------------------------
def expect_no_merge(orig_blocks, simp_blocks):
    return len(simp_blocks) == len(orig_blocks)

def expect_simple_merge(orig_blocks, simp_blocks):
    return len(simp_blocks) == 1

def expect_two_pass_merge(orig_blocks, simp_blocks):
    return len(simp_blocks) == 1

def expect_ignore_class0(orig_blocks, simp_blocks):
    return len(simp_blocks) == len(orig_blocks)

def expect_indentation_merge(orig_blocks, simp_blocks):
    return len(simp_blocks) == 1

def expect_floating_behavior(orig_blocks, simp_blocks):
    return len(simp_blocks) == len(orig_blocks)

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test suite for rule merging.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default=42)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    start = datetime.now(UTC).isoformat()
    print(f"[{start}] Verifying Decision Tree rule merging and simplification.")
    print(f"[INFO] Using seed: {args.seed}")

    run_test_case("Test 1 - No complementarity", generate_test1, expect_no_merge)
    run_test_case("Test 2 - Simple complementarity", generate_test2, expect_simple_merge)
    run_test_case("Test 3 - Deep complementarity", generate_test3, expect_simple_merge)
    run_test_case("Test 4 - Multi-level complementarity", generate_test4, expect_two_pass_merge)
    run_test_case("Test 5 - Floating precision", generate_test5, expect_floating_behavior)
    run_test_case("Test 6 - Indentation-insensitive", generate_test6, expect_indentation_merge)


if __name__ == "__main__":
    main()
