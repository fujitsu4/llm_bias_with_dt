"""
prepare_dataset_snli.py
Author: Zakaria JOUILIL

Description:
    Prepare the SNLI dataset by:
        - Cleaning text (whitespace + unicode normalization)
        - Removing sentences < 8 words
        - Removing sentences with semicolons
        - Keeping only sentences with exactly one syntactic ROOT
        - Selecting the first 2500 valid sentences

Inputs:
    - None (dataset is downloaded automatically from HuggingFace)

Outputs:
    - data/cleaned/snli_filtered.csv
    - logs/rejected_snli.txt   # rejected sentences (one per line)

Usage:
    python prepare_dataset_snli.py --output data/cleaned/snli_filtered.csv --target 2500
"""

from datasets import load_dataset
import pandas as pd
import spacy
import re

TARGET = 2500
MIN_WORDS = 8
OUTPUT = args.output

print("[INFO] Loading SNLI dataset...")
ds = load_dataset("snli", split="train")

nlp = spacy.load("en_core_web_sm")

sentences = []

# ----------------------------------------------------------
# 1) FUNCTION: clean a sentence BEFORE ANY processing
# ----------------------------------------------------------
def clean_sentence(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.strip()

    # remove tabs, non-breaking spaces, weird unicode spaces
    text = text.replace("\t", " ").replace("\xa0", " ")

    # collapse multiple spaces into a single one
    text = re.sub(r"\s+", " ", text)

    return text


# ----------------------------------------------------------
# 2) Collect all premises + hypotheses
# ----------------------------------------------------------
for row in ds:
    for field in ["premise", "hypothesis"]:
        text = row[field]
        if text:
            text = clean_sentence(text)
            if len(text.split()) >= MIN_WORDS:
                sentences.append(text)

print(f"[INFO] Collected raw sentences : {len(sentences)}")

# ----------------------------------------------------------
# 3) Remove duplicates
# ----------------------------------------------------------
unique_sentences = list(dict.fromkeys(sentences))
print(f"[INFO] After deduplication : {len(unique_sentences)}")


# ----------------------------------------------------------
# 4) Select more than target
# ----------------------------------------------------------
selected = unique_sentences[:TARGET * 2]


# ----------------------------------------------------------
# 5) Helper functions
# ----------------------------------------------------------
def has_unique_root(sentence: str) -> bool:
    doc = nlp(sentence)
    roots = [tok for tok in doc if tok.dep_ == "ROOT"]
    return len(roots) == 1

def contains_semicolon(sentence: str) -> bool:
    return ";" in sentence


# ----------------------------------------------------------
# 6) Filter sentences
# ----------------------------------------------------------
final_sentences = []
rejected_sentences = []

for s in selected:
    if contains_semicolon(s):
        rejected_sentences.append("[SEMICOLON] " + s)
        continue

    if has_unique_root(s):
        final_sentences.append(s)
        if len(final_sentences) >= TARGET:
            break
    else:
        rejected_sentences.append("[ROOT] " + s)

rejected_path = get_project_path("logs", "rejected_snli.txt")

with open(rejected_path, "w", encoding="utf-8") as f:
    for r in rejected_sentences:
        f.write(f"{r}\n")

print(f"[INFO] Valid sentences after filtering : {len(final_sentences)}")
print(f"[INFO] Rejected sentences              : {len(rejected_sentences)}")
print(f"[INFO] Saving rejected sentences to    : {rejected_path}")

# ----------------------------------------------------------
# 7) Build dataframe
# ----------------------------------------------------------
df = pd.DataFrame([
    {
        "sentence_id": i,
        "sentence": s,
        "dataset": "snli"
    }
    for i, s in enumerate(final_sentences)
])

df.to_csv(OUTPUT, sep=";", index=False)
print(f"[INFO] Saving cleaned dataset to : {len(final_sentences)}")

# ----------------------------------------------------------
# 8) CLI execution
# ----------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare SNLI dataset")
    parser.add_argument("--output", type=str, default="data/cleaned/snli_filtered.csv",
                        help="Output CSV path")
    parser.add_argument("--target", type=int, default=2500,
                        help="Number of sentences to keep")
    parser.add_argument("--min_words", type=int, default=8,
                        help="Minimum number of words per sentence")
    parser.add_argument("--verbose", action="store_true",
                        help="Print rejected sentences")

    args = parser.parse_args()

    # Display rejected sentences for verbose mode
    if args.verbose and rejected_sentences:
        print("\n=== FULL LIST OF REJECTED SENTENCES ===")
        for r in rejected_sentences:
            print("-", r)
        print("=== END OF REJECTED SENTENCES ===\n")
