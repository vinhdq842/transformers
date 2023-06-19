import torch
from torch import nn
from modules.layers import PositionalEncoding, Embedding
from modules.blocks import EncoderBlock, DecoderBlock


class Encoder(nn.Module):
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
        super(Encoder, self).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_emb = PositionalEncoding(max_length, d_model)
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(n_heads, d_model, d_ff, d_k, d_v, p_drop)
                for _ in range(n_blocks)
            ]
        )
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor):
        x = self.dropout(self.positional_emb(self.embedding(x)))

        for block in self.encoder_blocks:
            x = block(x, x_mask)

        return x


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
        p_drop: float,
    ):
        super(
            Decoder,
            self,
        ).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_emb = PositionalEncoding(max_length, d_model)
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(n_heads, d_model, d_ff, d_k, d_v, p_drop)
                for _ in range(n_blocks)
            ]
        )
        self.dropout = nn.Dropout(p=p_drop)

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
