import sys
from typing import Callable, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch import nn

sys.path.append("../../")
from models import GPT2, GPT2Args
from utils.datasets import TextDatasetForCausalLM
from utils.helpers import count_params, create_pad_mask, create_subsequent_mask
from utils.trainer import TrainingArgs, eval, train_and_val
from utils.tokenizers import ByteLevelBPETokenizer

torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = ByteLevelBPETokenizer()
tokenizer.load_state_dict(torch.load("tokenizers/bbpe-20k.pth"))

model_args = GPT2Args(
    block_size=1024,
    vocab_size=len(tokenizer),
    n_layers=12,
    n_heads=12,
    n_embed=768,
    dropout=0.0,
    bias=False,
)


def loss_batch_causal(
    batch: Tuple[torch.Tensor], model: nn.Module, device: str, loss_fn: Callable
):
    src, src_lens, tgt = (
        batch[0].to(device),
        batch[1].to(device),
        batch[2].to(device),
    )

    src_mask = create_subsequent_mask(src_lens, create_pad_mask(src_lens))
    logits = model(src, src_mask)

    return loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1)), {}

train_data = np.fromfile("datasets/train.bin")
val_data = np.fromfile("datasets/val.bin")
batch_size = 32

training_args = TrainingArgs(
    loss_batch=lambda batch, model, device: loss_batch_causal(
        batch,
        model,
        device,
        nn.CrossEntropyLoss(ignore_index=tokenizer.special_tokens[tokenizer.PAD]),
    ),
    train_dl=DataLoader(
        TextDatasetForCausalLM(train_data,model_args.block_size),
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        collate_fn=tp_collate,
    ),
    val_dl=DataLoader(
        TextDatasetForCausalLM(val_data,model_args.block_size),
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        collate_fn=tp_collate,
    ),
    n_epochs=5,
    scheduler_step_per="step",
    n_accum_steps=8,
    n_warmup_steps=5000,
    device=device,    
)

def init_weights(module:nn.Module):    
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

model = GPT2(model_args).to(device)
model.apply(init_weights)

count_params(model)

