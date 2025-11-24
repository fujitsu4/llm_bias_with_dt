"""
merge_datasets.py
Author: Zakaria JOUILIL

Description:
    Exhaustive verification between:
        - Merged datasets (merged_datasets.csv)
        - Initial SpaCy features (spacy_features.csv)
Checks performed (printed to log):
A. Completeness / alignment
    1) Same sentence IDs after the merge (no missing lines)
    2) No empty lines in features (no SpaCy incoherence)
    3) No empty fields for key columns (no SpaCy incoherence)

B. Feature-by-feature checks (by token), relying on spaCy:
    1) Token text consistency (sequence)
    2) is_numeric -> token.like_num
    3) is_punctuation <-> token.is_punct and token.pos_ == "PUNCT"
    4) is_stopword <-> token.is_stop
    5) is_pronoun <-> token.pos_ == "PRON" OR token.lower in pronoun_list
    6) pos <-> pos_id consistency (unique mapping)
    7) dep_label <-> dep_id consistency
    8) ent_label <-> ent_id consistency
    9) depth_in_tree and num_dependents are recomposed and must match

Inputs:
    - Merged datasets (merged_datasets.csv)
    - Initial SpaCy features (spacy_features.csv)
    
Outputs:
    - spacy_summary.txt (summary and error samples (max 20 per test))
    
Usage:
    python -m src.spacy.verify_spacy_features --sentences_csv data/cleaned/merged_datasets.csv --features_csv outputs/spacy/spacy_features.csv --log_file logs/spacy_logs.txt
"""

import argparse
import pandas as pd
import spacy
from collections import defaultdict
from tqdm import tqdm
import sys

# -------------------------
# Helpers
# -------------------------
PRONOUNS = {
    # subject/object/reflexive/general pronouns (lowercase)
    "i","me","you","he","him","she","her","it","we","us","they","them",
    "mine","yours","his","hers","ours","theirs",
    "my","your","his","her","its","our","their",
    "myself","yourself","himself","herself","itself","ourselves","yourselves","themselves",
    "somebody","someone","something","some","anyone","anything","anybody",
    "nobody","none","noone","everyone","everybody","everything","each","either","neither",
    "both","another","other","others","which","what","who"
}

def compute_depth_and_children(doc):
    """
    Precompute depth and number of children for each token in a spaCy doc.
    Returns two lists aligned with token indices.
    depth: distance to head (ROOT) following heads until self
    num_children: len(list(token.children))
    """
    depths = []
    num_children = []
    for tok in doc:
        # compute depth (safe loop)
        depth = 0
        cur = tok
        visited = set()
        while cur.head is not cur and cur not in visited:
            visited.add(cur)
            depth += 1
            cur = cur.head
            if depth > 200:
                # safety
                break
        depths.append(depth)
        num_children.append(len(list(tok.children)))
    return depths, num_children

# -------------------------
# Main verification
# -------------------------
def verify(sentences_csv, features_csv, log_file, max_report=20):
    # ---- Setup logger (tee: print to console + file)
    log_f = open(log_file, "w")
    def log(msg):
        print(msg)
        log_f.write(msg + "\n")

    log(f"\n[INFO] Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    log(f"\n[INFO] Reading inputs...")
    sent_df = pd.read_csv(sentences_csv, sep = ";")
    feat_df = pd.read_csv(features_csv, sep=";", keep_default_na=False)

    log(f"[INFO] Rows before cleaning (sentences): {len(sent_df)}")
    log(f"[INFO] Rows before cleaning (features) : {len(feat_df)}")

    # Normalize column names for robustness
    feat_df.columns = [c.strip() for c in feat_df.columns]
    sent_df.columns = [c.strip() for c in sent_df.columns]

    # Required columns
    required_sent_cols = {"sentence_id", "sentence"}
    required_feat_cols = {
        "sentence_id", "word_index", "word", "is_numeric", "is_punctuation",
        "is_stopword", "is_pronoun", "pos", "pos_id", "dep_label", "dep_id",
        "ent_label", "ent_id", "depth_in_tree", "num_dependents", "dataset"
    }

    missing_sent_cols = required_sent_cols - set(sent_df.columns)
    if missing_sent_cols:
        log(f"[FATAL] Sentences CSV is missing columns: {missing_sent_cols}")
        sys.exit(1)

    missing_feat_cols = required_feat_cols - set(feat_df.columns)
    if missing_feat_cols:
        log(f"[FATAL] Features CSV is missing columns: {missing_feat_cols}")
        sys.exit(1)

    # Make sure types are consistent
    sent_df['sentence_id'] = sent_df['sentence_id'].astype(int)
    feat_df['sentence_id'] = feat_df['sentence_id'].astype(int)
    feat_df['word_index'] = feat_df['word_index'].astype(int)

    # A. Completeness checks
    log("\n [INFO] A. Completeness / alignment checks")

    # A1) same set of sentence ids (no sentence dropped)
    sent_ids = set(sent_df['sentence_id'].tolist())
    feat_sent_ids = set(feat_df['sentence_id'].unique().tolist())
    missing_in_features = sent_ids - feat_sent_ids
    extra_in_features = feat_sent_ids - sent_ids

    log(f"[INFO] Total sentences in sentences_csv: {len(sent_ids)}")
    log(f"[INFO] Total unique sentence_id in features_csv: {len(feat_sent_ids)}")
    if missing_in_features:
        log(f"[ERROR] {len(missing_in_features)} sentences present in sentences_csv but missing in features_csv.")
        log(" [ERROR] Sample missing ids:", list(sorted(list(missing_in_features)))[:max_report])
    else:
        log("[INFO] ✓ All sentence ids from sentences_csv are present in features_csv")

    if extra_in_features:
        log(f"[WARNING] {len(extra_in_features)} sentence_id present in features_csv but not in sentences_csv.")
        log(" [WARNING] Sample extra ids:", list(sorted(list(extra_in_features)))[:max_report])
    else:
        log("[INFO] ✓ No extra sentence ids in features_csv")

    # A2) no completely empty rows in features CSV (all fields empty)
    empty_rows = feat_df[feat_df.isnull().all(axis=1)]
    if len(empty_rows) > 0:
        log(f"[ERROR] Found {len(empty_rows)} completely empty rows in features CSV")
        display_sample = empty_rows.head(max_report)
        log(display_sample.to_string(index=False))
    else:
        log("[INFO] ✓ No completely empty rows in features CSV")

    # A3) no empty values in critical columns
    critical_cols = ["sentence_id","word_index","word","pos","dep_label","ent_label","depth_in_tree","num_dependents"]
    empty_values = {}
    for col in critical_cols:
        nulls = feat_df[feat_df[col].isnull() | (feat_df[col].astype(str).str.strip() == "")]
        if len(nulls) > 0:
            empty_values[col] = len(nulls)
    if empty_values:
        log("[ERROR] Found empty values in critical columns:")
        for k,v in empty_values.items():
            log(f" - {k}: {v} empty")
    else:
        log("[INFO] ✓ No empty values in critical feature columns")

    # B. Feature-level validation
    log("\n[INFO] B. Feature-level validation (recomputing via spaCy where needed)")

    # Build mappings consistency checks (pos -> pos_id, dep_label -> dep_id, ent_label -> ent_id)
    log("\n[INFO] Consistency checks for id mappings (pos/dep/ent) ...")
    def check_consistency(label_col, id_col, name):
        mapping = {}
        inconsistent = {}
        for lab, gid in feat_df[[label_col, id_col]].itertuples(index=False, name=None):
            # convert to str for safe matching
            lab_s = str(lab)
            gid_i = int(gid) if (pd.notna(gid) and str(gid).strip() != "") else None
            if lab_s not in mapping:
                mapping[lab_s] = gid_i
            else:
                if mapping[lab_s] != gid_i:
                    inconsistent.setdefault(lab_s, set()).update({mapping[lab_s], gid_i})
        if inconsistent:
            log(f"[ERROR] Inconsistent {name} -> id mapping for {len(inconsistent)} labels. Sample:")
            cnt = 0
            for lab, ids in inconsistent.items():
                log(f"  - {lab}: ids={sorted(list(ids))}")
                cnt += 1
                if cnt >= max_report:
                    break
        else:
            log(f"[INFO] ✓ {name} -> id mapping is consistent (one unique id per label).")
        return mapping

    pos_map = check_consistency("pos", "pos_id", "POS")
    dep_map = check_consistency("dep_label", "dep_id", "Dep label")
    ent_map = check_consistency("ent_label", "ent_id", "Ent label")

    # Now per-token checks using spaCy by iterating sentence by sentence
    log("\n[INFO] Per-token checks using spaCy (token sequence, pos/punct/stop/pronoun/num/depth/children)...")
    token_sequence_mismatch = []
    numeric_issues = []
    punct_issues = []
    stop_issues = []
    pronoun_issues = []
    depth_issues = []
    child_issues = []
    pos_id_mismatches = []
    dep_id_mismatches = []
    ent_id_mismatches = []
    token_missing_in_sentences = []

    # Build a dict to access features by (sentence_id -> list ordered by word_index)
    feats_by_sent = {}
    for sid, group in feat_df.groupby('sentence_id'):
        # sort by word_index
        group_sorted = group.sort_values('word_index')
        feats_by_sent[int(sid)] = list(group_sorted.to_dict(orient='records'))

    # Iterate sentences (only those present in feats_by_sent)
    for _, srow in tqdm(sent_df.iterrows(), total=sent_df.shape[0]):
        sid = int(srow['sentence_id'])
        sentence = srow['sentence']
        if sid not in feats_by_sent:
            token_missing_in_sentences.append(sid)
            continue
        feat_tokens = feats_by_sent[sid]
        # Re-tokenize with spaCy
        doc = nlp(sentence)
        doc_tokens = [tok.text for tok in doc]
        # Quick token count check already done; still ensure ordering
        if len(doc_tokens) != len(feat_tokens):
            # record mismatch and continue deeper comparison still (we'll compare available)
            token_sequence_mismatch.append((sid, len(doc_tokens), len(feat_tokens), doc_tokens[:10], [ft['word'] for ft in feat_tokens[:10]]))
        # compute depth and num_children with spaCy
        depths, children_counts = compute_depth_and_children(doc)
        # iterate token by token (index based)
        for i, ft in enumerate(feat_tokens):
            reported_word = str(ft['word'])
            reported_word_index = int(ft['word_index'])
            # If spaCy shorter, skip out-of-range checks
            if i >= len(doc):
                # more tokens in features than spaCy
                token_sequence_mismatch.append((sid, "feature_longer", i, reported_word))
                continue
            sp_token = doc[i]
            sp_text = sp_token.text
            # 1) token text equality (exact)
            if sp_text != reported_word:
                token_sequence_mismatch.append((sid, i, sp_text, reported_word))

            # 2) is_numeric: verify with spaCy token.like_num and POS == NUM
            reported_is_num = int(ft.get('is_numeric', 0))
            sp_like_num = bool(sp_token.like_num)
            sp_pos = sp_token.pos_
            sp_dep = sp_token.dep_
            if reported_is_num == 1:
                # must be recognized as number by spaCy
                if not sp_like_num:
                    numeric_issues.append((sid, i, reported_word, reported_is_num, sp_like_num, sp_pos))

            # 3) is_punctuation: if reported 1 then sp_token.is_punct True; and if sp_token.pos_ == 'PUNCT' then reported must be 1
            reported_is_punct = int(ft.get('is_punctuation', 0))
            if reported_is_punct == 1 and not sp_token.is_punct:
                punct_issues.append((sid, i, reported_word, reported_is_punct, sp_token.is_punct, sp_token.pos_))
            if sp_token.pos_ == "PUNCT" and reported_is_punct != 1:
                punct_issues.append((sid, i, reported_word, reported_is_punct, sp_token.is_punct, sp_token.pos_))

            # 4) is_stopword
            reported_stop = int(ft.get('is_stopword', 0))
            if reported_stop != int(sp_token.is_stop):
                stop_issues.append((sid, i, reported_word, reported_stop, sp_token.is_stop))

            # 5) is_pronoun: if reported==1 then sp_token.pos_ should be PRON or token.lower in PRONOUNS; also if sp_token.pos_ == PRON then reported should be 1
            reported_pron = int(ft.get('is_pronoun', 0))
            lowerw = sp_token.text.lower()
            sp_is_pron = (sp_token.pos_ == "PRON") or (lowerw in PRONOUNS)
            if reported_pron == 1 and not sp_is_pron:
                pronoun_issues.append((sid, i, reported_word, reported_pron, sp_token.pos_, lowerw))
            if sp_token.pos_ == "PRON" and reported_pron != 1:
                pronoun_issues.append((sid, i, reported_word, reported_pron, sp_token.pos_, lowerw))

            # 6) pos <-> pos_id consistency already checked globally; verify reported pos matches spaCy
            reported_pos = str(ft.get('pos', ''))
            if reported_pos != sp_pos:
                pos_id_mismatches.append((sid, i, reported_word, reported_pos, sp_pos))

            # 7) dep_label <-> dep_id consistency checked globally; verify reported dep_label matches spaCy
            reported_dep = str(ft.get('dep_label', ''))
            if reported_dep != sp_dep:
                dep_id_mismatches.append((sid, i, reported_word, reported_dep, sp_dep))

            # 8) ent_label: compare reported to sp_token.ent_type_ (empty -> "NONE")
            reported_ent = str(ft.get('ent_label', 'NONE') if pd.notna(ft.get('ent_label')) else "NONE")
            sp_ent = sp_token.ent_type_ if sp_token.ent_type_ else "NONE"
            if reported_ent != sp_ent:
                ent_id_mismatches.append((sid, i, reported_word, reported_ent, sp_ent))

            # 9) depth_in_tree & num_dependents
            reported_depth = int(ft.get('depth_in_tree', -1))
            reported_children = int(ft.get('num_dependents', -1))
            sp_depth = depths[i]
            sp_children = children_counts[i]
            if reported_depth != sp_depth:
                depth_issues.append((sid, i, reported_word, reported_depth, sp_depth))
            if reported_children != sp_children:
                child_issues.append((sid, i, reported_word, reported_children, sp_children))

    # Print aggregated results
    def report_list(name, lst):
        if not lst:
            log(f"✓ {name} OK")
        else:
            print(f"[ERROR] {len(lst)} issues for {name}. Sample (<= {max_report}):")
            for item in lst[:max_report]:
                log("  ", item)

    log("\n=== RESULTS SUMMARY ===")
    report_list("Token sequence mismatches", token_sequence_mismatch)
    report_list("Numeric issues (is_numeric)", numeric_issues)
    report_list("Punctuation issues (is_punctuation)", punct_issues)
    report_list("Stopword issues (is_stopword)", stop_issues)
    report_list("Pronoun issues (is_pronoun)", pronoun_issues)
    report_list("POS mismatches (reported pos vs spaCy pos)", pos_id_mismatches)
    report_list("Dep-label mismatches (reported dep vs spaCy dep)", dep_id_mismatches)
    report_list("Ent-label mismatches (reported ent vs spaCy ent)", ent_id_mismatches)
    report_list("Depth mismatches (depth_in_tree)", depth_issues)
    report_list("Children mismatches (num_dependents)", child_issues)

    log("\n=== VERDICT ===")
    total_issues = sum([
        len(token_sequence_mismatch), len(numeric_issues), len(punct_issues),
        len(stop_issues), len(pronoun_issues), len(pos_id_mismatches),
        len(dep_id_mismatches), len(ent_id_mismatches), len(depth_issues),
        len(child_issues)
    ])
    if total_issues == 0:
        log("ALL TESTS PASSED — features file appears consistent with spaCy and the sentences file.")
    else:
        log(f"[ERROR] Total flagged issues across tests: {total_issues}. Inspect samples above.")
    log_f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify spacy features CSV against sentences CSV")
    parser.add_argument("--sentences_csv", type=str, required=True, help="CSV with sentences (sentence_id, sentence, ...)")
    parser.add_argument("--features_csv", type=str, required=True, help="CSV with token-level spacy features")
    parser.add_argument("--log_file", type=str, required=True,
                    help="Output text file where all logs will be saved")
    args = parser.parse_args()

    verify(args.sentences_csv, args.features_csv, log_file=args.log_file)
