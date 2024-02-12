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


@dataclass
class VanillaTransformerArgs:
    vocab_size: int = 32000
    n_heads: int = 8
    n_blocks: int = 6
    d_model: int = 512
    d_ff: int = 512 * 4
    d_k: int = 64
    d_v: int = 64
    max_length: int = 512
    p_drop: float = 0.1
    bias: bool = True  # bias in Linears and LayerNorms
    weight_tying: bool = True


class EncoderBlock(nn.Module):
    def __init__(self, args: VanillaTransformerArgs):
        super(EncoderBlock, self).__init__()
        self.args = args
        self.self_attn = MultiHeadAttention(
            args.n_heads, args.d_model, args.d_k, args.d_v
        )
        self.norm1 = LayerNorm(args.d_model, args.bias)
        self.ffn = PointwiseFeedForward(args.d_model, args.d_ff, args.bias)
        self.norm2 = LayerNorm(args.d_model, args.bias)
        self.dropout = nn.Dropout(p=args.p_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class DecoderBlock(nn.Module):
    def __init__(self, args: VanillaTransformerArgs):
        super(DecoderBlock, self).__init__()
        self.causal_self_attn = MultiHeadAttention(
            args.n_heads, args.d_model, args.d_k, args.d_v
        )
        self.norm1 = LayerNorm(args.d_model, args.bias)
        self.cross_attn = MultiHeadAttention(
            args.n_heads, args.d_model, args.d_k, args.d_v
        )
        self.norm2 = LayerNorm(args.d_model, args.bias)
        self.ffn = PointwiseFeedForward(args.d_model, args.d_ff, args.bias)
        self.norm3 = LayerNorm(args.d_model, args.bias)
        self.dropout = nn.Dropout(p=args.p_drop)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ):
        x = self.norm1(x + self.dropout(self.causal_self_attn(x, x, x, x_mask)))
        x = self.norm2(
            x
            + self.dropout(
                self.cross_attn(x, encoder_outputs, encoder_outputs, src_mask)
            )
        )
        return self.norm3(x + self.dropout(self.ffn(x)))


class Transformer(nn.Module):
    def __init__(self, args: VanillaTransformerArgs):
        super(Transformer, self).__init__()
        self.args = args
        self.embedding = Embedding(args.vocab_size, args.d_model)
        self.pos_embedding = PositionalEncoding(args.max_length, args.d_model)

        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(args) for _ in range(args.n_blocks)]
        )
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(args) for _ in range(args.n_blocks)]
        )

        self.dropout = nn.Dropout(p=args.p_drop)
        self.fc = nn.Linear(args.d_model, args.vocab_size)
        # weight tying
        if args.weight_tying:
            self.fc.weight = self.embedding.emb.weight

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        src = self.dropout(self.pos_embedding(self.embedding(src)))

        for encoder in self.encoder_blocks:
            src = encoder(src, src_mask)

        return src

    def decode(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ):
        tgt = self.dropout(self.pos_embedding(self.embedding(tgt)))

        for decoder in self.decoder_blocks:
            tgt = decoder(tgt, tgt_mask, encoder_outputs, src_mask)

        return tgt

    def generate(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ):
        return self.fc(self.decode(tgt, tgt_mask, encoder_outputs, src_mask))

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ):
        return self.fc(self.decode(tgt, tgt_mask, self.encode(src, src_mask), src_mask))
