import re
import warnings
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple, Union

from tqdm import tqdm


class Tokenizer:
    r"""Base class for tokenizers."""

    PAD = "<pad>"
    SOS = "<sos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self):
        self.special_tokens = {
            Tokenizer.PAD: 0,
            Tokenizer.SOS: 1,
            Tokenizer.EOS: 2,
            Tokenizer.UNK: 3,
        }
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self.vocab = {}

    def register_special_tokens(self, special_tokens: Dict[str, int]):
        r"""Add new special tokens."""
        self.special_tokens.update(special_tokens)
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

    def build(
        self, raw_corpus: List[str], target_size: int = 256, verbose: bool = False
    ):
        r"""Train the tokenizer on the given corpus."""
        raise NotImplementedError

    def encode(self, text: str) -> List[int]:
        r"""Encode a trivial text."""
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        r"""Decode a list of token ids."""
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, Any]):
        raise NotImplementedError

    def __call__(
        self, text: Union[str, Iterable[str]]
    ) -> Union[List[int], List[List[str]]]:
        if single := isinstance(text, str):
            text = [text]

        ids = []
        for c in text:
            ids.append(self.encode(c))

        return ids[0] if single else ids

    def __len__(self) -> int:
        return len(self.vocab) + len(self.special_tokens)


class BPETokenizer(Tokenizer):
    r"""Byte-Pair Encoding for building shared vocabulary."""

    def __init__(
        self,
        lower: bool = False,
        split_pattern: str = r"\s?[^\s]+|\s*[\r\n]|\s+(?!\S)|\s+",
    ):
        super().__init__()
        self.lower = lower
        self.split_pattern = split_pattern
        self.merges = {}
        self.inverse_vocab = {}

    def _merge_pair(
        self, pair: Tuple[str, str], splits: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        r"""Reflect the merge onto the splits."""
        for chunk, split in splits.items():
            i = 0
            while i < len(split) - 1:
                if (split[i], split[i + 1]) == pair:
                    split = split[:i] + [split[i] + split[i + 1]] + split[i + 2 :]
                else:
                    i += 1
            splits[chunk] = split
        return splits

    def _compute_pair_freqs(
        self, chunk_freqs: Dict[str, int], splits: Dict[str, List[str]]
    ) -> Dict[str, int]:
        r"""Compute pair frequencies."""
        pair_freqs = defaultdict(int)
        for chunk, freq in chunk_freqs.items():
            split = splits[chunk]
            for p1, p2 in zip(split, split[1:]):
                pair_freqs[(p1, p2)] += freq

        return pair_freqs

    def _prepair(self, raw_text: str) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
        r"""Split the text into separated chunks and turn each chunk into characters."""
        chunk_freqs = defaultdict(int)
        splits = {}
        for chunk in re.findall(self.split_pattern, raw_text):
            chunk_freqs[chunk] += 1

            if chunk in splits:
                continue

            splits[chunk] = list(chunk)

        return chunk_freqs, splits

    def build(
        self, raw_corpus: List[str], target_size: int = 256, verbose: bool = False
    ):
        assert target_size > len(self)

        raw_text = "\n".join(raw_corpus)
        if self.lower:
            raw_text = raw_text.lower()

        chunk_freqs, splits = self._prepair(raw_text)

        for c in list(set("".join(chunk_freqs.keys())))[:target_size]:
            self.vocab[len(self)] = c

        pbar = tqdm(
            desc="Building vocabulary...",
            initial=len(self),
            total=target_size,
            disable=not verbose,
        )

        while len(self) < target_size:
            pair_freqs = self._compute_pair_freqs(chunk_freqs, splits)

            if len(pair_freqs):
                best_pair = max(pair_freqs, key=pair_freqs.get)
                self.merges[best_pair] = len(self)
                self.vocab[len(self)] = best_pair[0] + best_pair[1]
                splits = self._merge_pair(best_pair, splits)
                pbar.update(1)
            else:
                warnings.warn("Not enough pair to merge. Stopping...")
                break

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def _encode_chunk(self, chars: List[str]) -> List[int]:
        r"""Do the actual encoding."""
        while len(chars) >= 2:
            stats = defaultdict(int)
            for p1, p2 in zip(chars, chars[1:]):
                stats[(p1, p2)] += 1

            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break

            idx = 0
            while idx < len(chars) - 1:
                if (chars[idx], chars[idx + 1]) == pair:
                    chars = (
                        chars[:idx] + [chars[idx] + chars[idx + 1]] + chars[idx + 2 :]
                    )
                else:
                    idx += 1

        return [self.inverse_vocab.get(c, self.special_tokens[self.UNK]) for c in chars]

    def _encode_ordinary(self, text: str) -> List[int]:
        r"""Encode a text chunk that is not supposed to contain special tokens."""
        chunks = re.findall(self.split_pattern, text)
        ids = []
        for chunk in chunks:
            ids.extend(self._encode_chunk(list(chunk)))

        return ids

    def encode(self, text: str) -> List[int]:
        if self.lower:
            text = text.lower()

        special_pattern = (
            "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
        )
        chunks = re.split(special_pattern, text)

        ids = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                ids.append(self.special_tokens[chunk])
            else:
                ids.extend(self._encode_ordinary(chunk))

        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = []
        for id in ids:
            if id in self.inverse_special_tokens:
                tokens.append(self.inverse_special_tokens[id])
            elif id in self.vocab:
                tokens.append(self.vocab[id])
            else:
                raise ValueError(f"Invalid token id: {id}")

        return "".join(tokens)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.__class__.__name__,
            "merges": self.merges,
            "vocab": self.vocab,
            "lower": self.lower,
            "special_tokens": self.special_tokens,
            "split_pattern": self.split_pattern,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.merges = state_dict["merges"]
        self.vocab = state_dict["vocab"]
        self.lower = state_dict["lower"]
        self.special_tokens = state_dict["special_tokens"]
        self.split_pattern = state_dict["split_pattern"]

        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}


class ByteLevelBPETokenizer(Tokenizer):
    r"""Byte-level Byte-Pair Encoding Tokenizer for shared vocabulary."""

    def __init__(self):
        super().__init__()

    def build(
        self, raw_corpus: List[str], target_size: int = 256, verbose: bool = False
    ):
        raise NotImplementedError

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, Any]):
        raise NotImplementedError
