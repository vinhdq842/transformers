from dataclasses import dataclass
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple, Union

import torch
from torch import nn
from tqdm import tqdm


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

    mask = torch.tril(torch.ones(max_length, max_length) == 1).unsqueeze(0).unsqueeze(0)

    if pad_mask is not None:
        mask = mask.to(pad_mask.device) & pad_mask

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


class BPETokenizer:
    r"""Byte-Pair Encoding for building shared vocabulary."""

    pad = "<pad>"
    sos = "<sos>"
    eos = "<eos>"
    unk = "<unk>"

    def __init__(
        self,
        raw_corpus: List[str] = None,
        target_size: int = 100,
        lower: bool = False,
    ):
        self.merges = {}
        self.target_size = target_size
        self.lower = lower
        self.special_tokens = [
            BPETokenizer.pad,
            BPETokenizer.sos,
            BPETokenizer.eos,
            BPETokenizer.unk,
        ]

        self.vocab = self.special_tokens
        self._st2i = {tk: i for i, tk in enumerate(self.vocab)}
        self._i2st = {i: tk for tk, i in self._st2i.items()}

        if raw_corpus is not None:
            self.build(raw_corpus)

    def _compute_token_freqs(self, corpus: List[List[str]]):
        freqs = defaultdict(int)
        for text in corpus:
            for token in text:
                freqs[token] += 1

        return freqs

    def _create_initial_vocab(self, token_freqs: Dict[str, int]):
        vocab = set()
        for token in token_freqs.keys():
            vocab.update(list(token))

        return self.special_tokens + sorted(list(vocab))

    def _pretokenize(self, raw_corpus: List[str]) -> List[List[str]]:
        res = []
        for text in raw_corpus:
            text = re.sub(r"\s+", " ", text).strip()
            if self.lower:
                text = text.lower()

            res.append(re.findall(r"\s?[^\s]+", text))

        return res

    def _best_pair(
        self, token_freqs: Dict[str, int], splits: Dict[str, List[str]]
    ) -> Tuple[str, str]:
        pair_freqs = defaultdict(int)
        max_freq = 0
        best_pair = (None, None)

        for token, freq in token_freqs.items():
            split = splits[token]
            if len(split) == 1:
                continue

            for i in range(len(split) - 1):
                pair_freqs[(split[i], split[i + 1])] += freq
                freq_ = pair_freqs[(split[i], split[i + 1])]
                if freq_ > max_freq:
                    max_freq = freq_
                    best_pair = (split[i], split[i + 1])

        return best_pair

    def _merge_pair(
        self, a: str, b: str, token_freqs: Dict[str, int], splits: Dict[str, List[str]]
    ):
        for token in token_freqs.keys():
            split = splits[token]

            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if (a, b) == (split[i], split[i + 1]):
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            splits[token] = split

        return splits

    def build(self, raw_corpus: List[str], target_size=None):
        corpus = self._pretokenize(raw_corpus)
        token_freqs = self._compute_token_freqs(corpus)
        self.vocab = self._create_initial_vocab(token_freqs)
        splits = {token: [st for st in token] for token in token_freqs.keys()}

        if target_size is None:
            target_size = self.target_size

        pbar = tqdm(
            desc="Building vocabulary...", initial=len(self.vocab), total=target_size
        )

        while len(self.vocab) < target_size:
            best_pair = self._best_pair(token_freqs, splits)
            if best_pair == (None, None):
                break
            self.vocab.append(best_pair[0] + best_pair[1])
            self.merges[best_pair] = best_pair[0] + best_pair[1]
            splits = self._merge_pair(*best_pair, token_freqs, splits)
            pbar.update(1)

        self._st2i = {tk: i for i, tk in enumerate(self.vocab)}
        self._i2st = {i: tk for tk, i in self._st2i.items()}

    def tokenize(self, input: Union[str, Iterable[str]]):
        if is_str := isinstance(input, str):
            input = [input]

        res = []
        input = self._pretokenize(input)

        for text in input:
            splits = [list(token) for token in text]

            for pair, merge in self.merges.items():
                for idx, split in enumerate(splits):
                    i = 0
                    while i < len(split) - 1:
                        if (split[i], split[i + 1]) == pair:
                            split = split[:i] + [merge] + split[i + 2 :]
                        else:
                            i += 1
                    splits[idx] = split

            res.append(
                [
                    tk if tk in self._st2i else BPETokenizer.unk
                    for tk in sum(
                        splits,
                        [],
                    )
                ]
            )

        return res[0] if is_str else res

    def encode(self, input: Union[List[List[str]], List[str]]):
        if is_single := isinstance(input, list) and (
            not len(input) or isinstance(input[0], str)
        ):
            input = [input]

        res = [
            [self._st2i.get(i, self._st2i[BPETokenizer.unk]) for i in seq]
            for seq in input
        ]

        return res[0] if is_single else res

    def decode(self, input: Union[List[List[int]], List[int]]):
        if is_single := isinstance(input, list) and (
            not len(input) or isinstance(input[0], int)
        ):
            input = [input]

        res = [[self._i2st.get(i, BPETokenizer.unk) for i in seq] for seq in input]

        return res[0] if is_single else res

    def state_dict(self):
        return {
            "merges": self.merges,
            "vocab": self.vocab,
            "st2i": self._st2i,
            "i2st": self._i2st,
            "target_size": self.target_size,
            "lower": self.lower,
        }

    def load_state_dict(self, state_dict):
        self.merges = state_dict["merges"]
        self.vocab = state_dict["vocab"]
        self._st2i = state_dict["st2i"]
        self._i2st = state_dict["i2st"]
        self.target_size = state_dict["target_size"]
        self.lower = state_dict["lower"]

    def __call__(self, input: Union[List[List[str]], List[str]]):
        return self.encode(self.tokenize(input))

    def __len__(self):
        return len(self.vocab)
