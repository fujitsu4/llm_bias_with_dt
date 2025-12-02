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
from transformers import BertConfig, BertModel


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

    model.config.output_attentions = True  # To be sure
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

    # ----------------------------------------------
    # Convert directly without retokenization
    # ----------------------------------------------
    input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(bert_tokens)])
    attention_mask = torch.ones_like(input_ids)

    # Add device
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Safety check
        if getattr(outputs, "attentions", None) is None:
            raise RuntimeError(
                "Model did not return attentions. "
                "Ensure model.config.output_attentions=True."
            )

        # outputs.attentions: tuple of (num_layers tensors)
        attentions = torch.stack(outputs.attentions, dim=0).squeeze(1)
        # shape: (num_layers, num_heads, seq_len, seq_len)

    num_layers, num_heads, seq_len, _ = attentions.shape

    # ---------------------------------------------------------
    # Compute layer sums
    # ---------------------------------------------------------
    layer_sums = []
    for L in range(num_layers):
        layer_sum = attentions[L].sum(dim=0).sum(dim=0)
        layer_sums.append(layer_sum)

    # ---------------------------------------------------------
    # Compute head-wise sums
    # ---------------------------------------------------------
    head_sums = []
    for L in range(num_layers):
        per_layer = []
        for H in range(num_heads):
            head_sum = attentions[L][H].sum(dim=0)
            per_layer.append(head_sum)
        head_sums.append(per_layer)

    return {
        "tokens": bert_tokens,
        "attentions": attentions,
        "layer_sums": layer_sums,
        "head_sums": head_sums
    }