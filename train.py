from typing import Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import NLLLoss

from config.config import Config
from tokenizer import Tokenizer
from model import Transformer
from spectrogram import MelSpectrogram
from data.dataset import OszDataset
from data.loader import OszLoader


class OsuTransformer(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.tokenizer = Tokenizer()
        self.transformer = Transformer(
            self.tokenizer.vocab_size(),
            self.tokenizer.pad_id,
            config.model.d_model,
            config.model.n_decoder_layer,
            config.model.n_encoder_layer,
            config.model.n_head,
            config.model.n_hidden,
            config.model.dropout,
        )
        self.spectrogram = MelSpectrogram(
            config.spectrogram.sample_rate,
            config.spectrogram.n_fft,
            config.spectrogram.n_mels,
            config.spectrogram.hop_length,
        )
        self.loss_fn = NLLLoss(ignore_index=self.tokenizer.pad_id, size_average=True)
        self.config = config

    def training_step(self, batch, batch_idx):
        target = batch["target"][:, 1:]
        labels = batch["target"][:, :-1]
        source = batch["source"]

        source = torch.flatten(source, start_dim=1)
        source = self.spectrogram.forward(source)
        logits = self.transformer.forward(source, target)
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        target = batch["target"][:, 1:]
        labels = batch["target"][:, :-1]
        source = batch["source"]

        source = torch.flatten(source, start_dim=1)
        source = self.spectrogram.forward(source)
        logits = self.transformer.forward(source, target)
        loss = self.loss_fn(logits, labels)
        self.log("val_loss", loss)

    def train_dataloader(self):
        loader = OszLoader(self.config.spectrogram.sample_rate, 0, 10, "max")
        dataset = OszDataset(
            self.config.dataset.train,
            loader,
            self.tokenizer,
            self.config.spectrogram.sample_rate,
            self.config.spectrogram.hop_length,
            self.config.dataset.src_seq_len,
            self.config.dataset.tgt_seq_len,
        )
        train_loader = DataLoader(dataset, 2, num_workers=0, pin_memory=True)
        return train_loader

    def configure_optimizers(self):
        optimizer = AdamW(self.transformer.parameters(), 1e-3)
        return optimizer


    config = Config()
    model = OsuTransformer(config)
trainer = pl.Trainer(precision=32, max_steps=10)
trainer.fit(model)
