import os
import sys

import torch

sys.path.append("../../")

from utils.tokenizers import ByteLevelBPETokenizer

text = open("datasets/raw_text.txt").read()

tokenizer = ByteLevelBPETokenizer()
tokenizer.build(
    [text],
    target_size=15000,
    verbose=True,
)
os.makedirs("tokenizers", exist_ok=True)
torch.save(tokenizer.state_dict(), "tokenizers/bbpe-15k.pth")
