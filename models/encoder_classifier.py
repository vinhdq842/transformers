from dataclasses import dataclass

import torch
from torch import nn

from modules.layers import (
    Embedding,
    LayerNorm,
    MultiHeadAttention,
    PointwiseFeedForward,
    PositionalEncoding,
)
from utils.helpers import create_pad_mask


@dataclass
class EncoderArgs:
    vocab_size: int = 32000
    n_heads: int = 8
    n_blocks: int = 6
    d_model: int = 512
    d_head: int = 64
    bias: bool = False
    p_drop: float = 0.1
    max_length: int = 512
    n_classes: int = 3


class EncoderBlock(nn.Module):
    def __init__(self, args: EncoderArgs):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            args.n_heads, args.d_model, args.d_head, args.d_head
        )
        self.norm1 = LayerNorm(args.d_model, args.bias)
        self.ffn = PointwiseFeedForward(args.d_model, args.d_model * 4, args.bias)
        self.norm2 = LayerNorm(args.d_model, args.bias)
        self.dropout = nn.Dropout(p=args.p_drop)

    def forward(self, x: torch.Tensor, mask: torch.tensor):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class EncoderClassifier(nn.Module):
    def __init__(self, args: EncoderArgs):
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
