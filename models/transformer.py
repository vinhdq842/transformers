import torch
from torch import nn

from modules.blocks import DecoderBlock, EncoderBlock
from modules.layers import Embedding, PositionalEncoding


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
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(max_length, d_model)

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(n_heads, d_model, d_ff, d_k, d_v, p_drop)
                for _ in range(n_blocks)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(n_heads, d_model, d_ff, d_k, d_v, p_drop)
                for _ in range(n_blocks)
            ]
        )

        self.dropout = nn.Dropout(p=p_drop)
        self.fc = nn.Linear(d_model, vocab_size)
        # weight tying
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
