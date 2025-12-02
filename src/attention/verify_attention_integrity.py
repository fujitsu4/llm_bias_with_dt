def test_attention_values_are_probabilities(data):
    att = data["attentions"]  # shape: (L, H, S, S)
    assert torch.all(att >= 0), "Attention contains negative values"
    assert torch.all(att <= 1 + 1e-6), "Attention contains values > 1"

def test_attention_rows_sum_to_one(data):
    att = data["attentions"]  # (L, H, S, S)

    # somme sur colonne (donc pour chaque "source token")
    row_sums = att.sum(dim=-1)

    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-4
    ), "Some attention rows do not sum to 1"

def test_matrix_matches_token_length(data, original_tokens):
    seq_len = len(original_tokens)
    att = data["attentions"]  # (L, H, S, S)
    assert att.shape[2] == seq_len, "Matrix row count != seq_len"
    assert att.shape[3] == seq_len, "Matrix col count != seq_len"

def test_tokens_preserved_order(data, original_tokens):
    returned_tokens = data["tokens"]
    assert returned_tokens == original_tokens, "Order of tokens changed!"

def test_attention_shape(data):
    att = data["attentions"]
    L, H, S1, S2 = att.shape

    assert L == 12, "Expected 12 layers for bert-base-cased"
    assert H == 12, "Expected 12 heads for bert-base-cased"
    assert S1 == S2, "Attention matrix must be square"



#!python -m src.attention.verify_attention_integrity --dir outputs/pretrained


import os
import argparse
import numpy as np
import pandas as pd


# ============================================================
#              Utility : load attention CSV cleanly
# ============================================================
def load_attention_csv(path):
    """
    Loads a CSV containing attention weights stored as semicolon-separated values
    in a single column or as a matrix. Ensures that everything becomes a float matrix.
    """
    df = pd.read_csv(path, sep=';')

    # Detect if stored as a single CSV matrix OR stringified list
    if df.shape[1] == 1:
        # One column = stringified rows
        matrix = df.iloc[:, 0].apply(lambda x: np.array(eval(x))).to_list()
        matrix = np.array(matrix, dtype=float)

    else:
        # Proper matrix shape (n x n)
        matrix = df.to_numpy(dtype=float)

    return matrix


# ============================================================
#                     Test 1: All values in [0,1]
# ============================================================
def test_value_range(att_matrix):
    if not (np.min(att_matrix) >= 0 and np.max(att_matrix) <= 1):
        raise AssertionError(
            f"[ERROR] Attention values not in [0,1]. Range = [{np.min(att_matrix)}, {np.max(att_matrix)}]"
        )


# ============================================================
#                     Test 2: Sum of rows ≈ 1
# ============================================================
def test_row_sums(att_matrix, tolerance=1e-3):
    row_sums = att_matrix.sum(axis=1)
    if not np.allclose(row_sums, np.ones_like(row_sums), atol=tolerance):
        raise AssertionError(
            f"[ERROR] Row sums not ≈ 1.\nExpected: 1\nGot: {row_sums}"
        )


# ============================================================
#            Test 3: Matrix size consistent with token count
# ============================================================
def test_square_matrix(att_matrix):
    if att_matrix.shape[0] != att_matrix.shape[1]:
        raise AssertionError(
            f"[ERROR] Attention matrix is not square: {att_matrix.shape}"
        )


# ============================================================
#          Test 4: matrix must be non-empty
# ============================================================
def test_non_empty(att_matrix):
    if att_matrix.size == 0:
        raise AssertionError("[ERROR] Empty attention matrix.")


# ============================================================
#  Test 5: no NaN / inf
# ============================================================
def test_no_nan_inf(att_matrix):
    if np.isnan(att_matrix).any() or np.isinf(att_matrix).any():
        raise AssertionError("[ERROR] Matrix contains NaN or Inf values.")


# ============================================================
#               Run all tests on a single file
# ============================================================
def run_all_tests(path):
    print(f"\n--- Testing {path} ---")

    att = load_attention_csv(path)

    test_non_empty(att)
    test_square_matrix(att)
    test_value_range(att)
    test_row_sums(att)
    test_no_nan_inf(att)

    print("[OK] All tests passed!")


# ============================================================
#         Recursively scan directory for CSV files
# ============================================================
def scan_directory(base_path):
    files = []

    for root, dirs, filenames in os.walk(base_path):
        for f in filenames:
            if f.endswith(".csv"):
                files.append(os.path.join(root, f))

    if len(files) == 0:
        print(f"[WARNING] No CSV files found in {base_path}")

    return files


# ============================================================
#                     CLI ENTRY POINT
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Verify attention integrity.")
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing attention CSV files (ex: outputs/pretrained OR outputs/untrained/seed_12)"
    )
    args = parser.parse_args()

    files = scan_directory(args.dir)

    for csv_path in files:
        try:
            run_all_tests(csv_path)
        except Exception as e:
            print(f"[FAILED] {csv_path}")
            print(str(e))


if __name__ == "__main__":
    main()
