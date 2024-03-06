import re
import warnings
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple, Union

from tqdm import tqdm


class Tokenizer:
    r"""Base class for tokenizers."""

    pad = "<pad>"
    sos = "<sos>"
    eos = "<eos>"
    unk = "<unk>"

    def __init__(self):
        self.special_tokens = {
            Tokenizer.pad: 0,
            Tokenizer.sos: 1,
            Tokenizer.eos: 2,
            Tokenizer.unk: 3,
        }
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self.vocab = {}

    def register_special_tokens(self, special_tokens: Dict[str, int]):
        self.special_tokens.update(special_tokens)
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

    def build(self, raw_corpus: List[str], target_size=None):
        raise NotImplementedError

    def encode(self, input: str):
        raise NotImplementedError

    def decode(self, input: List[int]):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, Any]):
        raise NotImplementedError

    def __call__(
        self, input: Union[str, Iterable[str]]
    ) -> Union[List[int], List[List[str]]]:
        if single := isinstance(input, str):
            input = [input]

        res = []
        for inp in input:
            res.append(self.encode(inp))

        return res[0] if single else res

    def __len__(self):
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

    def _compute_pair_freqs(self, text: List[str], freqs=None):
        if freqs is None:
            freqs = defaultdict(int)

        for p1, p2 in zip(text, text[1:]):
            freqs[(p1, p2)] += 1

        return freqs

    def _pretokenize(self, raw_text: str) -> List[List[str]]:
        r"""Split the text into separated chunks and turn each chunk into characters."""
        res = []
        for text in re.findall(self.split_pattern, raw_text):
            if self.lower:
                text = text.lower()

            res.append(list(text))

        return res

    def _merge_pair(self, pair: Tuple[str, str], chars: List[str]):
        i = 0
        while i < len(chars) - 1:
            if (chars[i], chars[i + 1]) == pair:
                chars = chars[:i] + [chars[i] + chars[i + 1]] + chars[i + 2 :]
            else:
                i += 1

        return chars

    def build(self, raw_corpus: List[str], target_size: int, verbose: bool = False):
        assert target_size > len(self)

        raw_text = "\n".join(raw_corpus)
        corpus = self._pretokenize(raw_text)

        for c in list(set(raw_text))[:target_size]:
            self.vocab[len(self)] = c

        pbar = tqdm(
            desc="Building vocabulary...",
            initial=len(self),
            total=target_size,
            disable=not verbose,
        )

        while len(self) < target_size:
            pair_freqs = defaultdict(int)
            for chunk in corpus:
                self._compute_pair_freqs(chunk, pair_freqs)
            if len(pair_freqs):
                best_pair = max(pair_freqs, key=pair_freqs.get)

                self.merges[best_pair] = len(self)
                self.vocab[len(self)] = best_pair[0] + best_pair[1]
                corpus = [self._merge_pair(best_pair, p) for p in corpus]
                pbar.update(1)
            else:
                warnings.warn("Not enough pair to merge. Stopping...")
                break

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def _encode_chunk(self, chars: List[str]) -> List[int]:
        while len(chars) >= 2:
            stats = self._compute_pair_freqs(chars)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break
            chars = self._merge_pair(pair, chars)

        return [self.inverse_vocab.get(c, self.special_tokens[self.unk]) for c in chars]

    def _encode(self, text: str) -> List[int]:
        chunks = re.findall(self.split_pattern, text)
        ids = []
        for chunk in chunks:
            ids.extend(self._encode_chunk(list(chunk)))

        return ids

    def encode(self, text: str) -> List[int]:
        special_pattern = (
            "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
        )
        chunks = re.split(special_pattern, text)

        ids = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                ids.append(self.special_tokens[chunk])
            else:
                ids.extend(self._encode(chunk))

        return ids

    def decode(self, input: List[int]) -> str:
        tokens = []
        for token in input:
            if token in self.inverse_special_tokens:
                tokens.append(self.inverse_special_tokens[token])
            elif token in self.vocab:
                tokens.append(self.vocab[token])
            else:
                raise ValueError(f"Invalid token id: {token}")

        return "".join(tokens)

    def state_dict(self):
        return {
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

    def __init__(self): ...

    def build(self, raw_corpus: List[str], target_size=None):
        raise NotImplementedError

    def encode(self, input: Union[List[List[str]], List[str]]):
        raise NotImplementedError

    def decode(self, input: Union[List[List[int]], List[int]]):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, Any]):
        raise NotImplementedError
