from collections import Counter
from typing import List, Union

import numpy as np
import torch
from torch import nn

from utils.tokenizers import Tokenizer


def create_pad_mask(lengths: torch.Tensor, max_length=None):
    r"""Create an attention mask for not attending to PAD tokens.

    Args:
        lengths (torch.Tensor): tensor of lengths, of shape (batch_size,).
        max_length (int, optional): maximum length. if None, set to lengths.max().

    Returns:
        torch.Tensor: mask of shape (batch_size, 1, 1, max_length).
    """
    if max_length is None:
        max_length = lengths.max()

    x = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)

    return (x.unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1).unsqueeze(1)


def create_subsequent_mask(lengths: torch.Tensor, max_length=None, pad_mask=None):
    r"""Create an attention mask for not attending to future tokens
    (and PAD tokens if a pad_mask is specified).

    Args:
        lengths (torch.Tensor): tensor of lengths, of shape (batch_size,).
        max_length (int, optional): maximum length. if None, set to lengths.max().
        pad_mask (torch.Tensor, optional): pad mask to combine.
        Defaults to None.

    Returns:
        torch.Tensor: mask of shape (batch_size, 1, max_length, max_length).
    """
    if max_length is None:
        max_length = lengths.max()

    mask = (
        torch.tril(torch.ones(max_length, max_length, device=lengths.device) == 1)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    if pad_mask is not None:
        mask = mask & pad_mask

    return mask


def count_params(model: nn.Module):
    total = 0
    trainable = 0

    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    print(f"Total: {total:,} parameters.")
    print(f"Trainable: {trainable:,} parameters.")

    return total, trainable


def bleu_score(
    candidates: Union[List, torch.Tensor],
    references: Union[List, torch.Tensor],
    tokenizer: Tokenizer,
) -> float:
    r"""Averaging the sentence level unigram-BLEU scores."""
    bleu = 0

    def one(candidate, reference):
        can = Counter(candidate)
        ref = Counter(reference)
        overlap = 0
        for c in can:
            overlap += min(can.get(c, 0), ref.get(c, 0))

        return (
            min(1, np.exp(1 - len(reference) / len(candidate)))
            * overlap
            / sum(can.values())
        )

    for candidate, reference in zip(candidates, references):
        if not isinstance(candidate, list) or not isinstance(reference, list):
            candidate, reference = candidate.tolist(), reference.tolist()
        if tokenizer.special_tokens[tokenizer.EOS] in candidate:
            candidate = candidate[
                : candidate.index(tokenizer.special_tokens[tokenizer.EOS]) + 1
            ]
        if tokenizer.special_tokens[tokenizer.EOS] in reference:
            reference = reference[
                : reference.index(tokenizer.special_tokens[tokenizer.EOS]) + 1
            ]

        bleu += one(candidate, reference)

    return bleu / len(candidates)


def translate_one_sentence(
    model: nn.Module,
    tokenizer: Tokenizer,
    sentence: str,
    max_tokens: int = 20,
    device: str = "cpu",
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k=-1,
) -> str:
    if do_sample:
        assert (
            0 < temperature <= 2.0
        ), f"`temperature` must be between 0 and 2.0, received {temperature}"

    src_ids = torch.tensor([tokenizer(sentence)], device=device)
    src_lens = torch.tensor([src_ids.shape[1]], dtype=torch.int64, device=device)
    src_mask = create_pad_mask(src_lens)

    tgt_ids = torch.tensor([[tokenizer.special_tokens[tokenizer.SOS]]], device=device)
    tgt_lens = torch.tensor([1], dtype=torch.int64, device=device)
    tgt_mask = create_subsequent_mask(tgt_lens, pad_mask=create_pad_mask(tgt_lens))

    with torch.inference_mode():
        encoder_outputs = model.encode(src_ids, src_mask)
        while (
            tgt_ids[0][-1] != tokenizer.special_tokens[tokenizer.EOS] and max_tokens > 0
        ):
            next = model.generate(tgt_ids, tgt_mask, encoder_outputs, src_mask)[:, -1]

            if do_sample:
                next = next / temperature
                if top_k > 0:
                    v, _ = torch.topk(next, min(top_k, next.size(-1)))
                    next[next < v[:, [-1]]] = float("-inf")
                next_id = next.softmax(-1).multinomial(num_samples=1)
            else:
                next_id = next.argmax(-1, keepdim=True)

            tgt_ids = torch.cat((tgt_ids, next_id), dim=-1)
            tgt_lens += 1
            tgt_mask = create_subsequent_mask(
                tgt_lens, pad_mask=create_pad_mask(tgt_lens)
            )
            max_tokens -= 1

    return tokenizer.decode(tgt_ids[0].tolist())
