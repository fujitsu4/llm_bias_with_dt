"""
tokenize_with_bert.py
Author: Zakaria JOUILIL

Description:
    Tokenize sentences using BERT, preserving the exact correspondence between BERT tokens and original words (word_id)
Inputs :
    - data/cleaned/merged_datasets.csv
    - outputs/spacy/spacy_features.csv (necessary to retrieve the exact word order)

Outputs :
    - bert/bert_tokens.csv
Usage:
    python -m src.bert.tokenize_with_bert \
    --sentences_csv data/cleaned/merged_datasets.csv \
    --spacy_csv outputs/spacy/spacy_features.csv \
    --output outputs/bert/bert_tokens.csv
"""

import argparse
import pandas as pd
from transformers import BertTokenizerFast
from tqdm import tqdm


def load_word_sequences(spacy_csv):
    """
    Reconstructs, for each sentence, the ordered list of spaCy words using word_index.
    Returns:
        dict[ sentence_id -> list[str] ]
    """
    df = pd.read_csv(spacy_csv, sep=";", keep_default_na=False, na_values=[])
    

    word_sequences = {}
    for sid, group in df.groupby("sentence_id"):
        group_sorted = group.sort_values("word_index")
        words = group_sorted["word"].tolist()
        word_sequences[sid] = words

    return word_sequences


def tokenize_sentences(sentences_csv, spacy_csv, output_csv):
    sentences_df = pd.read_csv(sentences_csv, sep=";", keep_default_na=False, na_values=[])
    
    word_sequences = load_word_sequences(spacy_csv)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    output_rows = []

    for _, row in tqdm(sentences_df.iterrows(), total=len(sentences_df)):
        sid = row["sentence_id"]
        dataset = row.get("dataset", "")
        
        if sid not in word_sequences:
            raise ValueError(f"[FATAL] sentence_id {sid} not in spacy csv !")

        words = word_sequences[sid]

        encoded = tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        word_ids = encoded.word_ids()

        for bert_index, (tok, wid) in enumerate(zip(tokens, word_ids)):
            output_rows.append({
                "sentence_id": sid,
                "bert_index": bert_index,
                "bert_token": tok,
                "word_id": wid,    # None for CLS/SEP → pandas → NaN
                "dataset": dataset
            })

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(output_csv, sep=";", index=False)
    print(f"[OK] Saving Tokenization saved in : {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences_csv", required=True,
                        help="CSV containing the input sentences")
    parser.add_argument("--spacy_csv", required=True,
                        help="CSV containing spaCy features ((necessary to retrieve the exact word order))")
    parser.add_argument("--output", required=True,
                        help="Output CSV to store BERT tokens")

    args = parser.parse_args()

    tokenize_sentences(args.sentences_csv, args.spacy_csv, args.output)


if __name__ == "__main__":
    main()
