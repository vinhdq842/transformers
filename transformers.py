import torch
from torch import nn
from modules.encoder import Encoder
from modules.decoder import Decoder


class Transformer(nn.Module):
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
        p_drop: float,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            vocab_size, n_heads, max_length, n_blocks, d_model, d_ff, d_k, d_v, p_drop
        )
        self.decoder = Decoder(
            vocab_size, n_heads, max_length, n_blocks, d_model, d_ff, d_k, d_v, p_drop
        )

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ):
        return self.fc(
            self.decoder(tgt, self.encoder(src, src_mask), src_mask, tgt_mask)
        )
