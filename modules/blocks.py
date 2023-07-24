import torch
from torch import nn

from modules.layers import LayerNorm, MultiHeadAttention, PointwiseFeedForward


class EncoderBlock(nn.Module):
    def __init__(
        self, n_heads: int, d_model: int, d_ff: int, d_k: int, d_v: int, p_drop: float
    ):
        super(EncoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        self.norm1 = LayerNorm(d_model)
        self.ffn = PointwiseFeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class DecoderBlock(nn.Module):
    def __init__(
        self, n_heads: int, d_model: int, d_ff: int, d_k: int, d_v: int, p_drop: float
    ):
        super(DecoderBlock, self).__init__()
        self.causal_self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        self.norm1 = LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        self.norm2 = LayerNorm(d_model)
        self.ffn = PointwiseFeedForward(d_model, d_ff)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=p_drop)

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
