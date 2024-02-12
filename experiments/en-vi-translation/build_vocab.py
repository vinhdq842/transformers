import os
import sys

import torch

sys.path.append("../../")
from datasets import load_from_disk

from utils.tokenizers import BPETokenizer

vi_en_dataset = load_from_disk("datasets/cleaned")

# this takes a long time to finish, ~ hours
tokenizer = BPETokenizer(
    vi_en_dataset["train"]["vi"]
    + vi_en_dataset["train"]["en"]
    + vi_en_dataset["test"]["vi"]
    + vi_en_dataset["test"]["en"],
    target_size=20000,
    lower=True,
)

os.makedirs("tokenizers", exist_ok=True)
torch.save(tokenizer.state_dict(), "tokenizers/bpe-20k.pth")
