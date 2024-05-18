# inspired by https://github.com/karpathy/nanoGPT/blob/master/model.py
from dataclasses import dataclass

import torch
from torch import nn

from modules.layers import LayerNorm, MultiHeadAttention


@dataclass
class GPT2Args:
    block_size: int = 1024
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    n_layers: int = 12
    n_heads: int = 12
    n_embed: int = 768
    dropout: float = 0.0
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )


class MLP(nn.Module):
    def __init__(self, args: GPT2Args):
        super().__init__()
        self.w_1 = nn.Linear(args.n_embed, 4 * args.n_embed, args.bias)
        self.gelu = nn.GELU()
        self.w_2 = nn.Linear(4 * args.n_embed, args.n_embed, args.bias)
        self.dropout = args.dropout

    def forward(self, x: torch.Tensor):
        return self.dropout(self.w_2(self.gelu(self.w_1(x))))


class Block(nn.Module):
    def __init__(self, args: GPT2Args):
        super().__init__()
        self.norm1 = LayerNorm(args.n_embed, args.bias)
        self.causal_self_attn = MultiHeadAttention(
            args.n_heads,
            args.n_embed,
            args.n_embed // args.n_heads,
            args.n_embed // args.n_heads,
        )
        self.norm2 = LayerNorm(args.n_embed, args.bias)
        self.mlp = MLP(args)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        n = self.norm1(x)
        x = x + self.causal_self_attn(n, n, n, mask)
        n = self.norm2(x)

        return x + self.mlp(n)


class GPT2(nn.Module):
    def __init__(self, args: GPT2Args):
        super().__init__()
        self.args = args
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(args.vocab_size, args.n_embed),
                wpe=nn.Embedding(args.vocab_size, args.n_embed),
                drop=nn.Dropout(args.dropout),
                h=nn.ModuleList([Block(args) for _ in range(args.n_layers)]),
                ln_f=LayerNorm(args.n_embed, args.bias),
            )
        )
        self.lm_head = nn.Linear(args.n_embed, args.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x_embed = self.transformer.wte(x)
        x_pos = self.transformer.wpe(
            torch.arange(0, x.size(1), dtype=torch.int64, device=x.device)
        )
        x = self.transformer.drop(x_embed + x_pos)

        for block in self.transformer.h:
            x = block(x, mask)

        x = self.transformer.ln_f(x)

        return self.lm_head(x)
