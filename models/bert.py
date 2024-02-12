from dataclasses import dataclass

import torch
from torch import nn


from modules.layers import Embedding, PositionalEncoding
from utils.helpers import create_pad_mask


@dataclass
class BertArgs:
    n_classes: int = 3


class EncoderBlock(nn.Module): ...


class BertForClassification(nn.Module):
    def __init__(self, args: BertArgs):
        super().__init__()
        self.max_length = args.max_length
        self.emb = Embedding(args.vocab_size, args.d_model)
        self.pos_emb = PositionalEncoding(args.max_length, args.d_model)
        self.encoders = nn.ModuleList(
            [EncoderBlock(args) for _ in range(args.n_blocks)]
        )
        self.fc = nn.Linear(args.d_model, args.n_classes)
        self.dropout = nn.Dropout(p=args.p_drop)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor):
        x_mask = create_pad_mask(x_lengths, self.max_length)
        x = self.dropout(self.pos_emb(self.emb(x)))

        for encoder in self.encoders:
            x = encoder(x, x_mask)
        x_mask = x_mask.squeeze().unsqueeze(-1)
        x = (torch.relu(x) * x_mask).sum(dim=1) / x_mask.squeeze().sum(-1, keepdim=True)
        return self.fc(self.dropout(x))
