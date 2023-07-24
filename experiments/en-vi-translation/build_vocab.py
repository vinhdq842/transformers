import re
import sys

import torch

sys.path.append("../../")
from datasets import load_from_disk

from modules.utils import BPETokenizer

vi_en_dataset = load_from_disk("datasets/processed")


def clean(batch):
    en = batch["en"].lower()
    vi = batch["vi"].lower()

    en = re.sub(r"\s+", " ", en).strip()
    en = " ".join(list(filter(lambda x: len(x), en.split())))
    batch["en"] = en

    vi = re.sub(r"\s+", " ", vi).strip()
    vi = " ".join(list(filter(lambda x: len(x), vi.split())))
    batch["vi"] = vi
    return batch


vi_en_dataset = vi_en_dataset.map(clean)

tokenizer = BPETokenizer(
    vi_en_dataset["train"]["vi"]
    + vi_en_dataset["train"]["en"]
    + vi_en_dataset["test"]["vi"]
    + vi_en_dataset["test"]["en"],
    target_size=32000,
    lower=True,
)

torch.save(tokenizer.state_dict(), "tokenizer.pth")
