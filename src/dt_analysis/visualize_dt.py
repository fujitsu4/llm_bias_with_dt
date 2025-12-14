"""
visualize_dt.py
Author: Zakaria JOUILIL

Description:
    Convert a sklearn decision tree exported as .txt (export_text)
    into a Graphviz visualization and save it as a PNG image.

Note :
    This is an illustrative visualization.
    It does NOT reconstruct the exact sklearn internal tree.

Input :
    --input : the FULL decision tree .txt file (containing class == 0 and 1), NOT the simplified tree

Output :
    The decision tree png

Usage :
    python -m src.dt_analysis.visualize_dt \
    --input outputs/decision_tree/pretrained/layer_01_rules_full.txt \
    --output outputs/dt_analysis/pretrained/layer_01_dt.png
"""

import argparse
import os
import re
import graphviz


# ---------------------------------------------------------
# Step 1: Extract ONLY the decision tree text
# ---------------------------------------------------------
def extract_tree_lines(txt_path):
    tree_lines = []
    started = False

    with open(txt_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.rstrip()

            # stop if stats section starts
            if "=== Per-leaf statistics" in line:
                break

            # detect real tree start
            if line.lstrip().startswith("|---"):
                started = True

            if started:
                tree_lines.append(line)

    if not tree_lines:
        raise ValueError("No decision tree found in file.")

    return tree_lines


# ---------------------------------------------------------
# Step 2: Clean formatting
# ---------------------------------------------------------
def clean_line(line):
    """
    - remove |--- and indentation
    - truncate floats to 2 decimals (NEVER round here because we have rules like
    "relative_position > 0.99512" it can never be > 1 if it's round !)
    """
    # remove tree ASCII structure
    line = re.sub(r"[| ]*--- ", "", line)

    # truncate  numbers to 2 decimals
    def truncate_float(match):
        value = float(match.group(0))
        truncated = int(value * 100) / 100
        return f"{truncated:.2f}"

    line = re.sub(r"\d+\.\d+", truncate_float, line)

    return line.strip()


# ---------------------------------------------------------
# Step 3: Build Graphviz tree
# ---------------------------------------------------------
def build_graph(tree_lines):
    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="TB", fontsize="10")

    node_id = 0
    stack = []  # (depth, node_id)

    for raw_line in tree_lines:
        depth = raw_line.count("|   ")
        label = clean_line(raw_line)

        current_id = f"n{node_id}"
        node_id += 1

        # coloring
        if "class:" in label:
            color = "lightcoral" if "class: 1" in label else "lightblue"
            dot.node(current_id, label, shape="box", style="filled", fillcolor=color)
        else:
            dot.node(current_id, label, shape="box", style="filled", fillcolor="lightgray")

        # connect to parent
        while stack and stack[-1][0] >= depth:
            stack.pop()

        if stack:
            parent_id = stack[-1][1]
            dot.edge(parent_id, current_id)

        stack.append((depth, current_id))

    return dot


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Decision tree .txt file")
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tree_lines = extract_tree_lines(args.input)
    dot = build_graph(tree_lines)

    output_path = args.output.replace(".png", "")
    dot.render(output_path, cleanup=True)

    print(f"[OK] Decision tree image saved to {args.output}")


if __name__ == "__main__":
    main()