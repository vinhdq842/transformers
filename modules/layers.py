import math

import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, bias: bool, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, x):
        return torch.layer_norm(x, [self.d_model], self.gamma, self.beta, self.eps)


class PointwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, bias: bool):
        super(PointwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w_2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.w_2(torch.relu(self.w_1(x)))


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(Embedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor):
        return self.emb(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, max_length: int, d_model: int):
        super(PositionalEncoding, self).__init__()
        pe = torch.empty(max_length, d_model)

        col = torch.exp(
            -torch.arange(0, d_model, 2).float() * math.log(10000) / d_model
        )
        row = torch.arange(max_length).float().unsqueeze(1)

        pe[:, 0::2] = torch.sin(row * col)
        pe[:, 1::2] = torch.cos(row * col)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        return x + self.pe[:, : x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, d_k: int, d_v: int):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * d_v, bias=False)

        self.w_o = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        r"""Compute multi-head attention.

        Args:
            q (torch.Tensor): batch_size x seq_len_q x d_model.
            k (torch.Tensor): batch_size x seq_len_k x d_model.
            v (torch.Tensor): batch_size x seq_len_k x d_model.
            mask (torch.Tensor): batch_size x 1 x 1 x seq_len_k
            or batch_size x 1 x seq_len_k x seq_len_k.

        Returns:
            torch.Tensor: batch_size x seq_length_q x d_model.
        """
        bs, seq_len_q, _ = q.size()
        _, seq_len_k, _ = k.size()

        q = (
            self.w_q(q).view(bs, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        )  # bs x n_heads x seq_len_q x d_k
        k = (
            self.w_k(k).view(bs, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        )  # bs x n_heads x seq_len_k x d_k
        v = (
            self.w_v(v).view(bs, seq_len_k, self.n_heads, self.d_v).transpose(1, 2)
        )  # bs x n_heads x seq_len_k x d_v

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.d_k
        )  # bs x n_heads x seq_len_q x seq_len_k

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_probs = attn_scores.softmax(dim=-1)
        attn_outputs = torch.matmul(attn_probs, v)  # bs x n_heads x seq_len_q x d_v

        return self.w_o(
            attn_outputs.transpose(1, 2)
            .contiguous()
            .view(bs, seq_len_q, self.n_heads * self.d_v)
        )
