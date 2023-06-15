import torch
from torch import nn
from modules.layers import Embedding, PositionalEncoding
from modules.blocks import DecoderBlock


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_heads: int,
        max_length: int,
        n_blocks: int,
        d_model: int,
        d_ff: int,
        d_k: int,
        d_v: int,
        dropout: float,
    ):
        super(
            Decoder,
            self,
        ).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_emb = PositionalEncoding(max_length, d_model)
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(n_heads, d_model, d_ff, d_k, d_v, dropout)
                for _ in range(n_blocks)
            ]
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ):
        x = self.dropout(self.positional_emb(self.embedding(x)))

        for block in self.decoder_blocks:
            x = block(x, encoder_output, src_mask, tgt_mask)

        return x
