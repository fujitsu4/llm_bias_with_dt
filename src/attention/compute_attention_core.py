"""
compute_attention_core.py
Author : Zakaria JOUILIL

Description:
    Core module that provides clean, testable functions for extracting
    attention matrices from a BERT model (pretrained or untrained).

    This file MUST NOT save files. It only returns data structures so that:
        - The final pipeline script can reuse it cleanly.
        - The debug script can test it deeply.

Output format returned by compute_attention():
    {
        "tokens"        : [list of bert tokens actually fed to the model]
        "attentions"    : 4D tensor (num_layers, num_heads, seq_len, seq_len)
        "layer_sums"    : list of tensors (one per layer) of shape (seq_len,)
                          = sum over all heads, and sum over all source tokens
        "head_sums"     : list (num_layers) of lists (num_heads) of tensors
                          each tensor = (seq_len,) sum of attention received
                                        by each token from this head
    }
"""

import torch
from transformers import BertTokenizer, BertConfig, BertModel


# ---------------------------------------------------------
# Fix random seed (for untrained models)
# ---------------------------------------------------------
def fix_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------
# Load model: pretrained or random
# ---------------------------------------------------------
def load_model(pretrained: bool, seed: int = None):
    if pretrained:
        model = BertModel.from_pretrained(
            "bert-base-cased",
            output_attentions=True
        )
    else:
        if seed is None:
            raise ValueError("Seed must be provided for untrained model.")
        fix_seed(seed)
        config = BertConfig.from_pretrained("bert-base-cased", output_attentions=True)
        model = BertModel(config)

    model.config.output_attentions = True
    model.eval()
    return model


# ---------------------------------------------------------
# Extract attention matrices and compute summaries
# ---------------------------------------------------------
def compute_attention(model, tokenizer, bert_tokens):
    """
    Input:
        model       : BERT model
        tokenizer   : corresponding tokenizer
        bert_tokens : list of BERT tokens (strings) in correct order

    Returns:
        Dict containing:
            tokens        : bert tokens fed to BERT
            attentions    : (num_layers, num_heads, seq_len, seq_len)
            layer_sums    : list of (seq_len,)  sum over all heads + sources
            head_sums     : [layer][head] -> (seq_len,)  sum over sources
    """
    # Prepare input
    enc = tokenizer(
        bert_tokens,
        is_split_into_words=True,
        return_tensors="pt",
        add_special_tokens=True
    )

    with torch.no_grad():
        outputs = model(**enc)
        # outputs.attentions -> tuple of num_layers tensors
        # each tensor: (batch=1, num_heads, seq_len, seq_len)
        attentions = torch.stack(outputs.attentions, dim=0).squeeze(1)
        # shape: (num_layers, num_heads, seq_len, seq_len)
        # Example: (12, 12, seq_len, seq_len)

    num_layers, num_heads, seq_len, _ = attentions.shape

    # ---------------------------------------------------------
    # Compute layer sums (sum over heads, sum over FROM tokens)
    # ---------------------------------------------------------
    layer_sums = []  # list of length num_layers, each tensor shape: (seq_len,)
    for L in range(num_layers):
        # attentions[L] = (num_heads, seq_len, seq_len)
        layer_sum = attentions[L].sum(dim=0).sum(dim=0)
        layer_sums.append(layer_sum)

    # ---------------------------------------------------------
    # Compute head-wise sums (sum over FROM tokens)
    # ---------------------------------------------------------
    head_sums = []
    for L in range(num_layers):
        per_layer = []
        for H in range(num_heads):
            head_sum = attentions[L][H].sum(dim=0)  # (seq_len,)
            per_layer.append(head_sum)
        head_sums.append(per_layer)

    # Retrieve the actual tokens BERT used
    decoded_tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

    return {
        "tokens": decoded_tokens,
        "attentions": attentions,
        "layer_sums": layer_sums,
        "head_sums": head_sums
    }