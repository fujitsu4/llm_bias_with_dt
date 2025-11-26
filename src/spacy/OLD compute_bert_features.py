"""
compute_bert_features.py
Author: Zakaria JOUILIL
Description:
    Tokenize sentences with a BERT tokenizer and compute token-level features
    (post-tokenization) + inherit spaCy word-level features for each sub-token.

Inputs:
    - data/cleaned/merged_datasets.csv
    - outputs/spacy/spacy_features.csv
    - outputs/spacy/spacy_label_maps.json

Outputs:
    - outputs/spacy/bert_features.csv

Usage:
    python -m src.spacy.compute_bert_features \
        --sentences data/cleaned/merged_datasets.csv \
        --spacy_features outputs/spacy/spacy_features.csv \
        --label_maps outputs/spacy/spacy_label_maps.json \
        --tokenizer bert-base-uncased \
        --output outputs/spacy/bert_features.csv
"""

import argparse
import json
from collections import Counter, defaultdict
from tqdm import tqdm
import pandas as pd
from src.utils.paths import get_project_path
from transformers import AutoTokenizer

# -------------------------
# CLI
# -------------------------
parser = argparse.ArgumentParser(description="Compute BERT token-level features and merge with spaCy word-level features")
parser.add_argument("--sentences", type=str, required=True, help="CSV with sentences (sentence_id,sentence,dataset)")
parser.add_argument("--spacy_features", type=str, required=True, help="CSV with spaCy word-level features")
parser.add_argument("--label_maps", type=str, required=True, help="JSON with pos/dep/ent mappings (from compute_spacy_features)")
parser.add_argument("--tokenizer", type=str, default="bert-base-uncased", help="HuggingFace tokenizer name (fast tokenizer recommended)")
parser.add_argument("--output", type=str, required=True, help="Output CSV path for bert features")
parser.add_argument("--max_sentences", type=int, default=None, help="Optional limit (for fast testing)")
args = parser.parse_args()

SENT_CSV = get_project_path(*args.sentences.split("/"))
SPACY_CSV = get_project_path(*args.spacy_features.split("/"))
LABEL_MAP_PATH = get_project_path(*args.label_maps.split("/"))
OUTPUT_CSV = get_project_path(*args.output.split("/"))

# -------------------------
# Load tokenizer
# -------------------------
print("[INFO] Loading tokenizer:", args.tokenizer)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

# -------------------------
# Load inputs
# -------------------------
print(f"[INFO] Reading sentences: {SENT_CSV}")
sent_df = pd.read_csv(SENT_CSV, sep=";")
if args.max_sentences:
    sent_df = sent_df.iloc[: args.max_sentences].copy()
print(f"[INFO] Sentences loaded: {len(sent_df)}")

print(f"[INFO] Reading spaCy features: {SPACY_CSV}")
spacy_df = pd.read_csv(SPACY_CSV, sep=";")
# group spaCy features per sentence_id as list ordered by word_index
spacy_by_sent = {}
for sid, group in spacy_df.groupby("sentence_id"):
    g = group.sort_values("word_index").to_dict(orient="records")
    spacy_by_sent[int(sid)] = g

# load label maps (pos2id etc.) if available
print(f"[INFO] Loading label maps: {LABEL_MAP_PATH}")
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_maps = json.load(f)
pos2id_map = label_maps.get("pos2id", {}) if label_maps else {}
dep2id_map = label_maps.get("dep2id", {}) if label_maps else {}
ent2id_map = label_maps.get("ent2id", {}) if label_maps else {}

# -------------------------
# Prepare outputs
# -------------------------
records = []

# -------------------------
# Iterate sentences
# -------------------------
print("[INFO] Tokenizing and merging features (this may take a while)...")
for _, row in tqdm(sent_df.iterrows(), total=sent_df.shape[0]):
    sid = int(row["sentence_id"])
    sentence = row["sentence"]
    dataset = row.get("dataset", "")

    # if no spaCy features for this sentence, skip (or create empties)
    word_level = spacy_by_sent.get(sid, [])
    # create sentence-level token frequency (BERT token strings)
    # we will compute after tokenization for each sentence

    # Tokenize with mapping to word ids
    # pass sentence split into words so tokenizer.word_ids() works reliably
    words = sentence.split()  # note: we cleaned sentences earlier
    enc = tokenizer(words, is_split_into_words=True, add_special_tokens=True)
    token_ids = enc["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    # word_ids: list aligned with tokens (None for special tokens)
    word_ids = enc.word_ids()

    # compute token-level frequency within this sentence (on token string)
    token_freq = Counter(tokens)

    # compute num_subtokens per original word
    # count tokens per word_id (ignore None)
    subtok_count_by_word = defaultdict(int)
    for wid in word_ids:
        if wid is not None:
            subtok_count_by_word[wid] += 1

    # iterate tokens
    n_tokens = len(tokens)
    # find indexes of non-special tokens to compute relative positions excluding specials if desired
    non_special_positions = [i for i, wid in enumerate(word_ids) if wid is not None]
    for ti, tok in enumerate(tokens):
        wid = word_ids[ti]  # None if special token
        is_special = 1 if wid is None else 0

        # basic token info
        is_subword = 0
        if tok.startswith("##"):   # BERT style subword marker
            is_subword = 1
            tok_norm = tok[2:]
        else:
            tok_norm = tok

        token_length = len(tok_norm)

        # position in sentence (bert index)
        position_in_sentence = ti
        # relative position normalized among non-special tokens (if any)
        if non_special_positions:
            try:
                rank_in_non_special = non_special_positions.index(ti)
                relative_position = rank_in_non_special / (len(non_special_positions) - 1) if len(non_special_positions) > 1 else 0.0
            except ValueError:
                # token is special -> position w.r.t. full tokens
                relative_position = ti / (n_tokens - 1) if n_tokens > 1 else 0.0
        else:
            relative_position = ti / (n_tokens - 1) if n_tokens > 1 else 0.0

        # token_sentence_frequency (on the token string itself)
        token_sentence_frequency = token_freq.get(tok, 0)

        # token_rank: vocabulary id (from tokenizer vocabulary) if available
        vocab = tokenizer.get_vocab()
        token_rank = vocab.get(tok, -1)

        # num_subtokens for this token: if wid is None -> -1 ; else number of subtokens for that word
        if wid is None:
            num_subtokens = -1
        else:
            num_subtokens = subtok_count_by_word.get(wid, 1)

        # inherit spaCy features if wid is not None
        if wid is not None and wid < len(word_level):
            wrec = word_level[wid]
            # these fields expected in spacy_features.csv; otherwise guard with defaults
            pos = wrec.get("pos", "NONE")
            pos_id = int(wrec.get("pos_id", -1)) if pd.notna(wrec.get("pos_id")) else -1
            dep_label = wrec.get("dep_label", "NONE")
            dep_id = int(wrec.get("dep_id", -1)) if pd.notna(wrec.get("dep_id")) else -1
            ent_label = wrec.get("ent_label", "NONE")
            ent_id = int(wrec.get("ent_id", -1)) if pd.notna(wrec.get("ent_id")) else ent2id_map.get("NONE", -1)
            depth_in_tree = int(wrec.get("depth_in_tree", -1)) if pd.notna(wrec.get("depth_in_tree")) else -1
            num_dependents = int(wrec.get("num_dependents", -1)) if pd.notna(wrec.get("num_dependents")) else -1
            is_numeric = int(wrec.get("is_numeric", 0))
            is_punctuation = int(wrec.get("is_punctuation", 0))
            is_stopword = int(wrec.get("is_stopword", 0))
            is_pronoun = int(wrec.get("is_pronoun", 0))
        else:
            # specials or missing mapping -> sentinel defaults
            pos = "SPECIAL"
            pos_id = -1
            dep_label = "SPECIAL"
            dep_id = -1
            ent_label = "NONE"
            ent_id = ent2id_map.get("NONE", -1)
            depth_in_tree = -1
            num_dependents = -1
            is_numeric = 0
            is_punctuation = 0
            is_stopword = 0
            is_pronoun = 0

        # prev_pos / next_pos: pos_id of prev/next BERT token (inherited from their originating word)
        # find previous token index with word_id not None (or allow specials if desired)
        prev_pos_id = -1
        next_pos_id = -1
        # previous
        j = ti - 1
        while j >= 0:
            jw = word_ids[j]
            if jw is not None and jw < len(word_level):
                prev_pos = word_level[jw].get("pos", None)
                prev_pos_id = int(word_level[jw].get("pos_id", -1)) if pd.notna(word_level[jw].get("pos_id")) else pos2id_map.get(prev_pos, -1)
                break
            j -= 1
        # next
        j = ti + 1
        while j < len(word_ids):
            jw = word_ids[j]
            if jw is not None and jw < len(word_level):
                next_pos = word_level[jw].get("pos", None)
                next_pos_id = int(word_level[jw].get("pos_id", -1)) if pd.notna(word_level[jw].get("pos_id")) else pos2id_map.get(next_pos, -1)
                break
            j += 1

        # token_burstiness: placeholder (-1) — compute later with global stats script if needed
        token_burstiness = -1

        # Save record
        records.append({
            "sentence_id": sid,
            "bert_index": ti,
            "token": tok,
            "is_special_token": is_special,
            "is_subword": is_subword,
            "token_length": token_length,
            "position_in_sentence": position_in_sentence,
            "relative_position": relative_position,
            "token_sentence_frequency": token_sentence_frequency,
            "token_rank": token_rank,
            "token_burstiness": token_burstiness,
            "num_subtokens": num_subtokens,
            # inherited spaCy
            "pos": pos,
            "pos_id": pos_id,
            "dep_label": dep_label,
            "dep_id": dep_id,
            "ent_label": ent_label,
            "ent_id": ent_id,
            "depth_in_tree": depth_in_tree,
            "num_dependents": num_dependents,
            "is_numeric": is_numeric,
            "is_punctuation": is_punctuation,
            "is_stopword": is_stopword,
            "is_pronoun": is_pronoun,
            # neighbors
            "prev_pos_id": prev_pos_id,
            "next_pos_id": next_pos_id,
            "dataset": dataset
        })

# -------------------------
# Save output
# -------------------------
print(f"[INFO] Writing {len(records)} token rows to {OUTPUT_CSV}")
out_df = pd.DataFrame(records)
out_df.to_csv(OUTPUT_CSV, sep=";", index=False)
print("[INFO] Done.")
