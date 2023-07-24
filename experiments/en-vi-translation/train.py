import os
import sys
from collections import Counter

import numpy as np
import torch
from datasets import load_from_disk
from torch import nn
from torch.nn.functional import pad
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append("../../")

from models.transformer import Transformer
from modules.utils import (
    BPETokenizer,
    count_params,
    create_pad_mask,
    create_subsequent_mask,
)

vi_en_ids = load_from_disk("datasets/processed_ids_splits")


class TextDataset(Dataset):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def __getitem__(self, index):
        return (
            self.src[index],
            len(self.src[index]),
            self.tgt[index],
            len(self.tgt[index]),
        )

    def __len__(self):
        return len(self.src)

    @classmethod
    def pad(cls, inputs, tgt=False):
        def pad_data(x, length):
            x_padded = pad(
                x,
                (0, length - x.shape[0]),
                mode="constant",
                value=tokenizer._st2i[tokenizer.pad],
            )
            return x_padded

        max_len = max((len(x) for x in inputs)) + tgt
        padded = torch.stack([pad_data(torch.LongTensor(x), max_len) for x in inputs])

        return padded

    @classmethod
    def collate_fn(cls, batch):
        src = []
        src_lens = []
        tgt = []
        tgt_lens = []
        for s, sl, t, tl in batch:
            src.append(s)
            src_lens.append(sl)
            tgt.append(t)
            tgt_lens.append(tl)
        return (
            cls.pad(src),
            torch.LongTensor(src_lens),
            cls.pad(tgt, tgt=True),
            torch.LongTensor(tgt_lens),
        )


max_length = 256

vi_en_ids = vi_en_ids.filter(
    lambda batch: 2 < len(batch["ids_vi"]) <= max_length
    and 1 <= len(batch["ids_en"]) <= max_length
)

batch_size = 32

train_dl = DataLoader(
    TextDataset(vi_en_ids["train"]["ids_en"], vi_en_ids["train"]["ids_vi"]),
    batch_size=batch_size,
    pin_memory=True,
    shuffle=True,
    collate_fn=TextDataset.collate_fn,
)

val_dl = DataLoader(
    TextDataset(vi_en_ids["val"]["ids_en"], vi_en_ids["val"]["ids_vi"]),
    batch_size=batch_size,
    pin_memory=True,
    shuffle=False,
    collate_fn=TextDataset.collate_fn,
)

tokenizer = BPETokenizer()
tokenizer.load_state_dict(torch.load("tokenizer.pth"))


def bleu_score(candidates, references, tokenizer=tokenizer):
    bleu = 0

    def one(candidate, reference):
        can = Counter(candidate)
        ref = Counter(reference)
        overlap = 0
        for c in can:
            overlap += min(can.get(c, 0), ref.get(c, 0))

        return (
            min(1, np.exp(1 - len(reference) / len(candidate)))
            * overlap
            / sum(can.values())
        )

    for candidate, reference in zip(candidates, references):
        if not isinstance(candidate, list) or not isinstance(reference, list):
            candidate, reference = candidate.tolist(), reference.tolist()
            if tokenizer._st2i[tokenizer.pad] in candidate:
                candidate = candidate[: candidate.index(tokenizer._st2i[tokenizer.pad])]
            if tokenizer._st2i[tokenizer.pad] in reference:
                reference = reference[: reference.index(tokenizer._st2i[tokenizer.pad])]
        bleu += one(candidate, reference)

    return bleu / len(candidates)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vocab_size = len(tokenizer)
n_heads = 4
n_blocks = 4
d_model = 128
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

optimizer = Adam(model.parameters(), lr=0.0003, betas=(0.98, 0.99), eps=1e-9)
scheduler = ExponentialLR(optimizer, 0.999**0.125)
loss_fn = nn.CrossEntropyLoss(
    label_smoothing=0.1, ignore_index=tokenizer._st2i[tokenizer.pad]
)


os.makedirs("checkpoints", exist_ok=True)


def decode(model, tokenizer, src, max_token=30):
    src_ids = torch.tensor([tokenizer(src)]).to(device)
    src_lens = torch.tensor([src_ids.shape[1]], dtype=torch.int64)
    src_mask = create_pad_mask(src_lens).to(device)

    tgt_ids = torch.tensor([[tokenizer._st2i[tokenizer.sos]]]).to(device)
    tgt_lens = torch.tensor([[1]], dtype=torch.int64)
    tgt_mask = create_subsequent_mask(tgt_lens, pad_mask=create_pad_mask(tgt_lens)).to(
        device
    )

    model.eval()
    with torch.inference_mode():
        encoder_outputs = model.encode(src_ids, src_mask)
        while tgt_ids[0][-1] != tokenizer._st2i[tokenizer.eos] and max_token > 0:
            next = model.generate(tgt_ids, tgt_mask, encoder_outputs, src_mask)
            tgt_ids = torch.cat((tgt_ids, next[:, -1].argmax(-1, keepdim=True)), dim=-1)
            tgt_lens += 1
            tgt_mask = create_subsequent_mask(
                tgt_lens, pad_mask=create_pad_mask(tgt_lens)
            ).to(device)
            max_token -= 1
    return tgt_ids


def train_and_eval(
    model,
    optimizer,
    loss_fn,
    scheduler,
    epochs,
    train_dl,
    val_dl,
    accum_steps=8,
    early_stopping=10,
    model_name="sample_model",
):
    p_bar = tqdm(total=len(train_dl))

    best_val_loss = 25042001
    patience = 0
    step = 0

    for epoch in range(epochs):
        train_loss = 0
        train_bleu = 0
        val_loss = 0
        val_bleu = 0

        model.train()
        for src, src_lens, tgt, tgt_lens in train_dl:
            src, src_lens, tgt, tgt_lens = (
                src.to(device),
                src_lens.to(device),
                tgt.to(device),
                tgt_lens.to(device),
            )

            src_mask = create_pad_mask(src_lens)
            tgt_0 = tgt[:, :-1]
            tgt_0_mask = create_subsequent_mask(
                tgt_lens, pad_mask=create_pad_mask(tgt_lens)
            )
            tgt_1 = tgt[:, 1:].contiguous().view(-1)

            logits = model(src, src_mask, tgt_0, tgt_0_mask)
            loss = loss_fn(logits.view(-1, vocab_size), tgt_1)

            train_loss += loss.item()
            train_bleu += bleu_score(logits.argmax(-1), tgt[:, 1:])

            if accum_steps > 1:
                loss /= accum_steps

            loss.backward()

            step += 1
            if accum_steps > 1:
                if step % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            p_bar.update(1)
        scheduler.step()

        model.eval()
        with torch.inference_mode():
            for src, src_lens, tgt, tgt_lens in val_dl:
                src, src_lens, tgt, tgt_lens = (
                    src.to(device),
                    src_lens.to(device),
                    tgt.to(device),
                    tgt_lens.to(device),
                )

                src_mask = create_pad_mask(src_lens)
                tgt_0 = tgt[:, :-1]
                tgt_0_mask = create_subsequent_mask(
                    tgt_lens, pad_mask=create_pad_mask(tgt_lens)
                )
                tgt_1 = tgt[:, 1:].contiguous().view(-1)

                logits = model(src, src_mask, tgt_0, tgt_0_mask)
                loss = loss_fn(logits.view(-1, vocab_size), tgt_1)

                val_loss += loss.item()
                val_bleu += bleu_score(logits.argmax(-1), tgt[:, 1:])

        train_loss /= len(train_dl)
        val_loss /= len(val_dl)
        train_bleu = train_bleu / len(train_dl)
        val_bleu = val_bleu / len(val_dl)

        print(
            f"Epoch {epoch+1}:\n\tTrain loss: {train_loss:.6f} - Train bleu: {train_bleu:.6f}\n\tVal loss: {val_loss:.6f} - Val bleu: {val_bleu:.6f}"
        )
        print(
            f'\t{"".join(tokenizer.decode(decode(model,tokenizer,"a single female will lay about up to eggs at a time , up to about in her lifetime .")[0].tolist()))}'
        )

        if early_stopping > 0:
            if val_loss > best_val_loss:
                patience += 1

                if patience >= early_stopping:
                    print(
                        f"\tStopped since val loss has not improved in the last {early_stopping} epochs..."
                    )
                    break
            else:
                torch.save(
                    model.state_dict(),
                    f"checkpoints/{model_name}.pth",
                )
                print(f"\tCheckpoint saved at epoch {epoch+1}...")
                patience = 0
                best_val_loss = val_loss

        p_bar.reset()


epochs = 3000
model_name = "translation-128-4-4"

train_and_eval(
    model,
    optimizer,
    loss_fn,
    scheduler,
    epochs,
    train_dl,
    val_dl,
    accum_steps=8,
    early_stopping=200,
    model_name=model_name,
)
