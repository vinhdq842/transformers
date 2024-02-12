import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Tuple, Union

import torch
from torch import nn
from torch.optim import optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainingArgs:
    loss_batch: Callable
    train_dl: DataLoader
    val_dl: DataLoader
    test_dl: Union[DataLoader, None] = None
    n_epochs: int = 1
    scheduler_step_per: Literal["epoch", "step"] = "epoch"
    n_accum_steps: int = 8
    early_stopping_patience: int = 10
    n_warmup_steps: int = 5000
    device: str = "cpu"


def train_and_val(
    model: nn.Module,
    optimizer: optimizer.Optimizer,
    scheduler: Union[LRScheduler, None],
    args: TrainingArgs,
    model_name: str = "sample_model",
    infer_one_sample: Union[Callable, None] = None,
) -> Tuple[Dict[str, Dict[str, List]], float]:
    os.makedirs("checkpoints", exist_ok=True)
    training_history = {
        "train": {"loss": [], "metrics": []},
        "val": {"loss": [], "metrics": []},
    }
    best_val_loss = 25042001
    patience = 0
    step = 0

    print("Training:")
    p_bar = tqdm(total=len(args.train_dl))
    for epoch in range(args.n_epochs):
        train_loss = 0
        val_loss = 0
        train_metrics = {}
        val_metrics = {}

        model.train()
        for batch in args.train_dl:
            loss, mt = args.loss_batch(batch, model, args.device)

            train_loss += loss.item()

            for k, v in mt.items():
                if k in train_metrics:
                    train_metrics[k] += v
                else:
                    train_metrics[k] = v

            if args.n_accum_steps > 1:
                loss /= args.n_accum_steps

            loss.backward()

            step += 1
            if args.n_accum_steps > 1:
                if step % args.n_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    if scheduler and args.scheduler_step_per == "step":
                        scheduler.step()
            else:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if scheduler and args.scheduler_step_per == "step":
                    scheduler.step()

            p_bar.update(1)

        if scheduler and args.scheduler_step_per == "epoch":
            scheduler.step()

        model.eval()
        with torch.inference_mode():
            for batch in args.val_dl:
                loss, mt = args.loss_batch(batch, model, args.device)

                val_loss += loss.item()

                for k, v in mt.items():
                    if k in val_metrics:
                        val_metrics[k] += v
                    else:
                        val_metrics[k] = v

        train_loss /= len(args.train_dl)
        val_loss /= len(args.val_dl)

        for k in train_metrics.keys():
            train_metrics[k] /= len(args.train_dl.dataset)
            val_metrics[k] /= len(args.val_dl.dataset)

        training_history["train"]["loss"].append(train_loss)
        training_history["train"]["metrics"].append(train_metrics)
        training_history["val"]["loss"].append(val_loss)
        training_history["val"]["metrics"].append(val_metrics)

        print(f"Epoch {epoch+1}:")
        print(
            f"\tTrain loss: {train_loss:.6f} | Train {' | Train '.join([f'{k}: {v:.6f}' for k,v in train_metrics.items()])}"
        )
        print(
            f"\tVal loss: {val_loss:.6f} | Val {' | Val '.join([f'{k}: {v:.6f}' for k,v in val_metrics.items()])}"
        )

        if infer_one_sample:
            print(f"\t{infer_one_sample(model)}")

        if args.early_stopping_patience > 0:
            if val_loss > best_val_loss:
                patience += 1

                if patience >= args.early_stopping_patience:
                    print(
                        f"\tStopped since val loss has not improved in the last {args.early_stopping_patience} epochs..."
                    )
                    break
            else:
                patience = 0
                best_val_loss = val_loss
                torch.save(
                    model.state_dict(),
                    f"checkpoints/{model_name}.pth",
                )
                print(f"\tCheckpoint saved at epoch {epoch+1}...")

        p_bar.reset()

    return training_history, best_val_loss


def eval(
    model: nn.Module, loss_batch: Callable, test_dl: DataLoader, device: str = "cpu"
) -> Dict[str, Dict[str, List]]:
    test_loss = 0
    test_metrics = {}

    model.eval()
    print("Evaluation:")
    with torch.inference_mode():
        for batch in test_dl:
            loss, mt = loss_batch(batch, model, device)
            test_loss += loss.item()
            for k, v in mt.items():
                if k in test_metrics:
                    test_metrics[k] += v
                else:
                    test_metrics[k] = v

    test_loss /= len(test_dl)

    for k in test_metrics.keys():
        test_metrics[k] /= len(test_dl.dataset)

    print(
        f"\tTest loss: {test_loss:.6f} | Test {' | Test '.join([f'{k}: {v:.6f}' for k,v in test_metrics.items()])}"
    )

    return {"test": {"loss": [test_loss], "metrics": [test_metrics]}}
