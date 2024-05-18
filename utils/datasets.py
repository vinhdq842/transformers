import torch
from torch.nn.functional import pad
from torch.utils.data import Dataset, IterableDataset


class TweetDataset(Dataset):
    UNK = "[UNK]"
    PAD = "[PAD]"

    def __init__(self, df, vocab, max_length):
        self.df = df
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        text = str(self.df.clean_text.values[index]).split()

        return (
            self.df.category.values[index],
            [self.vocab.get(w, self.vocab[TweetDataset.UNK]) for w in text],
            len(text),
        )

    def collate_fn(self, batch):
        labels = []
        ids = []
        lens = []

        for label, idx, ln in batch:
            labels.append(label)
            ids.append(
                idx[: self.max_length]
                + max(0, self.max_length - len(idx)) * [self.vocab[TweetDataset.PAD]]
            )
            lens.append(ln if ln <= self.max_length else self.max_length)

        return torch.LongTensor(labels), torch.LongTensor(ids), torch.LongTensor(lens)


class TextPairDataset(Dataset):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def __getitem__(self, index):
        return (
            self.src[index],
            len(self.src[index]),
            self.tgt[index],
            len(self.tgt[index]),
        )

    def __len__(self):
        return len(self.src)


class TextPairDataCollate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def pad(self, inputs, tgt=False):
        def pad_data(x, length):
            x_padded = pad(
                x,
                (0, length - x.shape[0]),
                mode="constant",
                value=self.tokenizer.special_tokens[self.tokenizer.PAD],
            )
            return x_padded

        max_len = max((len(x) for x in inputs)) + tgt
        padded = torch.stack([pad_data(torch.LongTensor(x), max_len) for x in inputs])

        return padded

    def __call__(self, batch):
        src = []
        src_lens = []
        tgt = []
        tgt_lens = []
        for s, sl, t, tl in batch:
            src.append(s)
            src_lens.append(sl)
            tgt.append(t)
            tgt_lens.append(tl)

        return (
            self.pad(src),
            torch.LongTensor(src_lens),
            self.pad(tgt, tgt=True),
            torch.LongTensor(tgt_lens),
        )


class TextDatasetForCausalLM(IterableDataset):
    def __init__(self, raw_ids: torch.Tensor, block_size: int):
        self.raw_ids = raw_ids
        self.block_size = block_size

    def gen_sample(self):
        idx = torch.randint(len(self.raw_ids) - self.block_size, (1,))
        x, y = (
            self.raw_ids[idx : idx + self.block_size],
            self.raw_ids[idx + 1 : idx + self.block_size + 1],
        )

        yield (x, y, len(x), len(y))

    def __iter__(self):
        return self.gen_sample()
