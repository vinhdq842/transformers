import sys
from typing import Callable, Tuple

import torch
from datasets import load_from_disk
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from transformers import get_linear_schedule_with_warmup

sys.path.append("../../")

from models import Transformer
from utils.datasets import TextPairDataCollate, TextPairDataset
from utils.engine import eval, train_and_val
from utils.helpers import (bleu_score, count_params, create_pad_mask,
                           create_subsequent_mask, translate_one_sentence)
from utils.tokenizers import BPETokenizer, Tokenizer

torch.manual_seed(42)

tokenizer = BPETokenizer()
tokenizer.load_state_dict(torch.load("tokenizers/bpe.pth"))


def loss_batch_seq2seq(
    batch: Tuple[torch.Tensor], model: nn.Module, device: str, loss_fn: Callable
):
    src, src_lens, tgt, tgt_lens = (
        batch[0].to(device),
        batch[1].to(device),
        batch[2].to(device),
        batch[3].to(device),
    )
    src_mask = create_pad_mask(src_lens)
    tgt_0 = tgt[:, :-1]
    tgt_0_mask = create_subsequent_mask(tgt_lens, pad_mask=create_pad_mask(tgt_lens))
    tgt_1 = tgt[:, 1:].contiguous().view(-1)

    logits = model(src, src_mask, tgt_0, tgt_0_mask)

    return loss_fn(logits.view(-1, logits.size(-1)), tgt_1), {
        "bleu": bleu_score(logits.argmax(-1), tgt[:, 1:], tokenizer) * len(batch)
    }


vi_en_ids = load_from_disk("datasets/processed_ids_splits")

max_length = 256

vi_en_ids = vi_en_ids.filter(
    lambda batch: 2 < len(batch["ids_vi"]) <= max_length
    and 1 <= len(batch["ids_en"]) <= max_length
)

tp_collate = TextPairDataCollate(tokenizer)

batch_size = 32

train_dl = DataLoader(
    TextPairDataset(vi_en_ids["train"]["ids_en"], vi_en_ids["train"]["ids_vi"]),
    batch_size=batch_size,
    pin_memory=True,
    shuffle=True,
    collate_fn=tp_collate,
)

val_dl = DataLoader(
    TextPairDataset(vi_en_ids["val"]["ids_en"], vi_en_ids["val"]["ids_vi"]),
    batch_size=batch_size,
    pin_memory=True,
    shuffle=False,
    collate_fn=tp_collate,
)

test_dl = DataLoader(
    TextPairDataset(vi_en_ids["test"]["ids_en"], vi_en_ids["test"]["ids_vi"]),
    batch_size=batch_size,
    pin_memory=True,
    shuffle=False,
    collate_fn=tp_collate,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vocab_size = len(tokenizer)
n_heads = 8
n_blocks = 6
d_model = 384
d_k = d_v = d_model // n_heads
d_ff = 4 * d_model
p_drop = 0.1

model = Transformer(
    vocab_size, n_heads, max_length, n_blocks, d_model, d_ff, d_k, d_v, p_drop
).to(device)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

count_params(model)

n_epochs = 1000
n_accum_steps = 8
optimizer = Adam(model.parameters(), lr=0.0004, betas=(0.98, 0.99), eps=1e-9)
n_warmup_steps = 5000
# scheduler = LambdaLR(
#    optimizer,
#    lr_lambda=lambda x: min(max(x, 1) ** -0.5, max(x, 1) * warmup_steps**-1.5),
# )
scheduler = get_linear_schedule_with_warmup(
    optimizer, n_warmup_steps, n_epochs * len(train_dl) / n_accum_steps
)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer._st2i[tokenizer.pad])

model_name = f"translation-{d_model}-{n_blocks}-{n_heads}"

training_history, best_val_loss = train_and_val(
    model,
    optimizer,
    lambda batch, model, device: loss_batch_seq2seq(batch, model, device, loss_fn),
    scheduler,
    n_epochs,
    train_dl,
    val_dl,
    n_accum_steps=8,
    model_name=model_name,
    infer_one_sample=lambda m: translate_one_sentence(
        m,
        tokenizer,
        device,
        "Once upon a day, he met her, but didn't know that the girl would change his entire life.",
    ),
    device=device,
    scheduler_step_per="step",
)

model.load_state_dict(torch.load(f"checkpoints/{model_name}.pth"))
eval_history = eval(
    model,
    lambda batch, model, device: loss_batch_seq2seq(batch, model, device, loss_fn),
    test_dl,
    device=device,
)
