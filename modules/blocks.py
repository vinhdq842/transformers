import torch
from torch import nn
from modules.layers import LayerNorm, PointwiseFeedForward
import math


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
        self.self_causal_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        self.norm1 = LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        self.norm2 = LayerNorm(d_model)
        self.ffn = PointwiseFeedForward(d_model, d_ff)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ):
        x = self.norm1(x + self.dropout(self.self_causal_attn(x, x, x, tgt_mask)))
        x = self.norm2(
            x
            + self.dropout(self.cross_attn(x, encoder_output, encoder_output, src_mask))
        )
        return self.norm3(x + self.dropout(self.ffn(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, d_k: int, d_v: int):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_heads * d_k)
        self.w_k = nn.Linear(d_model, n_heads * d_k)
        self.w_v = nn.Linear(d_model, n_heads * d_v)

        self.w_o = nn.Linear(n_heads * d_v, d_model)

    def forward(self, q, k, v, mask=None):
        r"""Compute multi-head attention

        Args:
            x (torch.Tensor): batch_size x seq_len x d_model
            mask (torch.Tensor): batch_size x 1 x 1 x seq_len
        """
        bs, seq_len, _ = q.size()

        q = (
            self.w_q(q).view(bs, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        )  # bs x n_heads x seq_len x d_k
        k = (
            self.w_k(k).view(bs, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        )  # bs x n_heads x seq_len x d_k
        v = (
            self.w_v(v).view(bs, seq_len, self.n_heads, self.d_v).transpose(1, 2)
        )  # bs x n_heads x seq_len x d_v

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = attn_scores.softmax(dim=-1)

        attn_outputs = torch.matmul(attn_probs, v)

        return self.w_o(
            attn_outputs.transpose(1, 2)
            .contiguous()
            .view(bs, seq_len, self.n_heads * self.d_v)
        )
