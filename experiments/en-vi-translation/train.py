import sys
from typing import Callable, Tuple

import torch
from datasets import load_from_disk
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from transformers import get_linear_schedule_with_warmup

sys.path.append("../../")

from models import Transformer, VanillaTransformerArgs
from utils.datasets import TextPairDataCollate, TextPairDataset
from utils.trainer import TrainingArgs, eval, train_and_val
from utils.helpers import (
    bleu_score,
    count_params,
    create_pad_mask,
    create_subsequent_mask,
    translate_one_sentence,
)
from utils.tokenizers import BPETokenizer

torch.manual_seed(42)

tokenizer = BPETokenizer()
tokenizer.load_state_dict(torch.load("tokenizers/bpe-20k.pth"))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_args = VanillaTransformerArgs(
    vocab_size=len(tokenizer),
    n_heads=4,
    n_blocks=4,
    d_model=128,
    d_ff=4 * 128,
    d_k=128 // 4,
    d_v=128 // 4,
    max_length=256,
    p_drop=0.1,
    bias=False,
)


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


vi_en_ids = load_from_disk("datasets/processed_ids_splits_20k")
vi_en_ids = vi_en_ids.filter(
    lambda batch: 2 < len(batch["ids_vi"]) <= model_args.max_length
    and 1 <= len(batch["ids_en"]) <= model_args.max_length
)

tp_collate = TextPairDataCollate(tokenizer)
batch_size = 32

training_args = TrainingArgs(
    loss_batch=lambda batch, model, device: loss_batch_seq2seq(
        batch,
        model,
        device,
        nn.CrossEntropyLoss(ignore_index=tokenizer._st2i[tokenizer.pad]),
    ),
    train_dl=DataLoader(
        TextPairDataset(vi_en_ids["train"]["ids_en"], vi_en_ids["train"]["ids_vi"]),
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        collate_fn=tp_collate,
    ),
    val_dl=DataLoader(
        TextPairDataset(vi_en_ids["val"]["ids_en"], vi_en_ids["val"]["ids_vi"]),
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        collate_fn=tp_collate,
    ),
    test_dl=DataLoader(
        TextPairDataset(vi_en_ids["test"]["ids_en"], vi_en_ids["test"]["ids_vi"]),
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        collate_fn=tp_collate,
    ),
    n_epochs=1000,
    scheduler_step_per="step",
    n_accum_steps=8,
    n_warmup_steps=5000,
    device=device,
)


model = Transformer(model_args).to(device)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

count_params(model)

n_epochs = 1000
n_accum_steps = 8
optimizer = Adam(model.parameters(), lr=0.0004, betas=(0.98, 0.99), eps=1e-9)
n_warmup_steps = 5000
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    training_args.n_warmup_steps,
    training_args.n_epochs * len(training_args.train_dl) / training_args.n_accum_steps,
)

model_name = f"translation-{model_args.d_model}-{model_args.n_blocks}-{model_args.n_heads}-{model_args.vocab_size//1000}k"

training_history, best_val_loss = train_and_val(
    model,
    optimizer,
    scheduler,
    training_args,
    model_name=model_name,
    infer_one_sample=lambda m: translate_one_sentence(
        m,
        tokenizer,
        device,
        "Once upon a day , he met her , but did n't know that the girl would change his entire life .",
    ),
)

model.load_state_dict(torch.load(f"checkpoints/{model_name}.pth"))
eval_history = eval(
    model,
    training_args.loss_batch,
    training_args.test_dl,
    device=device,
)
