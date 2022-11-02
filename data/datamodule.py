from typing import Optional

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from config.config import Config
from tokenizer import Tokenizer
from data.dataset import OszDataset
from data.loader import OszLoader


class OszDataModule(pl.LightningDataModule):
    def __init__(self, config: Config):
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
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            pin_memory=True,
        )
