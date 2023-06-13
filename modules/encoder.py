import torch
from torch import nn
from modules.layers import Embedding, PositionalEncoding
from modules.blocks import EncoderBlock


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_head: int,
        max_length: int,
        n_block: int,
        d_model: int,
        d_ff: int,
        d_k: int,
        d_v: int,
    ):
        super(Encoder, self).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_emb = PositionalEncoding(max_length, d_model)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(n_head, d_model, d_ff, d_k, d_v) for _ in range(n_block)]
        )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor):
        x = self.positional_emb(self.embedding(x))

        for block in self.encoder_blocks:
            x = block(x, x_mask)

        return x
