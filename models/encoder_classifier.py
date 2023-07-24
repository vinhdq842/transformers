import torch
from torch import nn

from modules.blocks import EncoderBlock
from modules.layers import Embedding, PositionalEncoding
from modules.utils import create_pad_mask


class EncoderClassifier(nn.Module):
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
        n_classes: int,
        p_drop: float,
    ):
        super().__init__()
        self.max_length = max_length
        self.emb = Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(max_length, d_model)
        self.encoders = nn.ModuleList(
            [
                EncoderBlock(n_heads, d_model, d_ff, d_k, d_v, p_drop)
                for _ in range(n_blocks)
            ]
        )
        self.fc = nn.Linear(d_model, n_classes)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor):
        x_mask = create_pad_mask(x_lengths, self.max_length)
        x = self.dropout(self.pos_emb(self.emb(x)))

        for encoder in self.encoders:
            x = encoder(x, x_mask)

        return self.fc(
            self.dropout((torch.relu(x) * x_mask.squeeze().unsqueeze(-1)).mean(dim=1))
        )
