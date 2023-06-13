import torch
from torch import nn
from modules.layers import LayerNorm, PointwiseFeedForward
import math


class EncoderBlock(nn.Module):
    def __init__(self, n_head: int, d_model: int, d_ff: int, d_k: int, d_v: int):
        super(EncoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.norm1 = LayerNorm(d_model)
        self.ffn = PointwiseFeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor):
        x = self.norm1(x + self.self_attention(x, x, x, x_mask))
        return self.norm2(x + self.ffn(x))


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int):
        ...


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_head * d_k)
        self.w_k = nn.Linear(d_model, n_head * d_k)
        self.w_v = nn.Linear(d_model, n_head * d_v)

        self.w_o = nn.Linear(n_head * d_v, d_model)

    def forward(self, q, k, v, mask=None):
        """Compute multi-head attention

        x - batch_size x seq_len x d_model
        mask - batch_size x 1 x 1 x seq_len
        """
        bs, seq_len, _ = q.shape

        q = (
            self.w_q(q).view(bs, seq_len, self.n_head, self.d_k).transpose(1, 2)
        )  # bs x n_head x seq_len x d_k
        k = (
            self.w_k(k).view(bs, seq_len, self.n_head, self.d_k).transpose(1, 2)
        )  # bs x n_head x seq_len x d_k
        v = (
            self.w_v(v).view(bs, seq_len, self.n_head, self.d_v).transpose(1, 2)
        )  # bs x n_head x seq_len x d_v

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = attn_scores.softmax(dim=-1)

        attn_outputs = torch.matmul(attn_probs, v)

        return self.w_o(
            attn_outputs.transpose(1, 2)
            .contiguous()
            .view(bs, seq_len, self.n_head * self.d_v)
        )
