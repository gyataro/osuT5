import torch
import numpy as np
from transformers import T5Config, Adafactor
from omegaconf import open_dict, DictConfig
from torch.optim import Optimizer
from torch.utils.data import DataLoader, get_worker_info
from torch.optim.lr_scheduler import (
    LRScheduler,
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
)

from osuT5.model import T5
from osuT5.dataset import OszDataset, OszLoader, OsuParser
from osuT5.tokenizer import Tokenizer


def get_config(args: DictConfig) -> T5Config:
    config = T5Config.from_pretrained(args.model.name)

    if hasattr(args.model, "overwrite"):
        for k, v in args.model.overwrite.items():
            assert hasattr(config, k), f"config does not have attribute {k}"
            setattr(config, k, v)

    if hasattr(args.model, "add_config"):
        for k, v in args.model.add_config.items():
            assert not hasattr(config, k), f"config already has attribute {k}"
            setattr(config, k, v)

    tokenizer = Tokenizer()
    setattr(config, "vocab_size", tokenizer.vocab_size())
    return config


def get_model(config: T5Config) -> T5:
    model = T5(config)
    return model


def get_tokenizer() -> Tokenizer:
    return Tokenizer()


def get_optimizer(model: T5, args: DictConfig) -> Optimizer:
    no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.optim.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = Adafactor(
        optimizer_grouped_parameters,
        lr=args.optim.base_lr,
        relative_step=False,
    )

    return optimizer


def get_scheduler(optimizer: Optimizer, args: DictConfig) -> LRScheduler:
    scheduler_p1 = LinearLR(
        optimizer,
        start_factor=0.5,
        end_factor=1,
        total_iters=args.optim.warmup_steps,
        last_epoch=-1,
    )

    scheduler_p2 = CosineAnnealingLR(
        optimizer,
        T_max=args.optim.total_steps - args.optim.warmup_steps,
        eta_min=args.optim.final_cosine,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_p1, scheduler_p2],
        milestones=[args.optim.warmup_steps],
    )

    return scheduler


def get_dataloaders(tokenizer: Tokenizer, args: DictConfig) -> dict[str, DataLoader]:
    loader = OszLoader(
        args.model.spectrogram.sample_rate,
        args.loader.min_difficulty,
        args.loader.max_difficulty,
        args.loader.mode,
    )
    parser = OsuParser()
    dataset = {
        "train": OszDataset(
            args.train_dataset_path,
            args.model.spectrogram.sample_rate,
            args.model.spectrogram.hop_length,
            args.model.max_seq_len,
            args.model.max_target_len,
            loader,
            parser,
            tokenizer,
        ),
        "test": OszDataset(
            args.test_dataset_path,
            args.model.spectrogram.sample_rate,
            args.model.spectrogram.hop_length,
            args.model.max_seq_len,
            args.model.max_target_len,
            loader,
            parser,
            tokenizer,
        ),
    }

    dataloaders = {}
    for split in ["train", "test"]:
        batch_size = args.optim.batch_size // args.optim.grad_acc

        dataloaders[split] = DataLoader(
            dataset[split],
            batch_size=batch_size,
            num_workers=args.dataloader.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
        )

    return dataloaders["train"], dataloaders["test"]


def worker_init_fn(worker_id: int) -> None:
    """
    Give each dataloader worker a unique seed.
    This ensures that each worker loads the .osz archives
    in a different sequential order.
    """
    worker_seed = get_worker_info().seed
    numpy_seed = (worker_id + worker_seed) % 2**32 - 1
    np.random.seed(numpy_seed)
