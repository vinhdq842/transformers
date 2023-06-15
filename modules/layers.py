import torch
import math
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        return torch.layer_norm(x, [self.d_model], self.gamma, self.beta, self.eps)


class PointwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super(PointwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

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
