from __future__ import annotations

from typing import Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, get_worker_info

from .tokenizer import Tokenizer
from .dataset import OszDataset
from .loader import OszLoader


class OszDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loader = OszLoader(
            config.spectrogram.sample_rate,
            config.osz_loader.min_difficulty,
            config.osz_loader.max_difficulty,
            config.osz_loader.mode,
        )
        self.tokenizer = Tokenizer()

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = OszDataset(
            self.config.dataset.train,
            self.loader,
            self.tokenizer,
            self.config.spectrogram.sample_rate,
            self.config.spectrogram.hop_length,
            self.config.dataset.src_seq_len,
            self.config.dataset.tgt_seq_len,
        )
        self.val_dataset = OszDataset(
            self.config.dataset.val,
            self.loader,
            self.tokenizer,
            self.config.spectrogram.sample_rate,
            self.config.spectrogram.hop_length,
            self.config.dataset.src_seq_len,
            self.config.dataset.tgt_seq_len,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
        )


def worker_init_fn(worker_id):
    """
    Give each dataloader worker a unique seed.
    This ensures that each worker loads the .osz archives
    in a different sequential order.
    """
    worker_seed = get_worker_info().seed
    numpy_seed = (worker_id + worker_seed) % 2**32 - 1
    np.random.seed(numpy_seed)
