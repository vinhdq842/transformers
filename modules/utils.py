import torch
from torch import nn


def create_sequence_mask(lengths: torch.Tensor, max_length=None):
    if max_length is None:
        max_length = lengths.max()
    x = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)
    return (x.unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1).unsqueeze(1)


def create_subsequent_mask(max_length):
    return torch.tril(torch.ones(max_length, max_length)).unsqueeze(0).unsqueeze(0)


def count_params(model: nn.Module):
    total = 0
    trainable = 0

    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    print(f"Total: {total:,} parameters.")
    print(f"Trainable: {trainable:,} parameters.")
