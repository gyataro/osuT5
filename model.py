from __future__ import annotations
from warnings import filterwarnings

import torch
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import MulticlassF1Score

from module.transformer import Transformer
from module.spectrogram import MelSpectrogram
from train.config import Config
from train.optimizer import Adafactor
from train.scheduler import get_cosine_schedule_with_warmup
from utils.tokenizer import Tokenizer

filterwarnings("ignore", ".*does not have many workers.*")


class OsuTransformer(pl.LightningModule):
    def __init__(self, config=Config()):
        super().__init__()
        self.save_hyperparameters(config)
        self.tokenizer = Tokenizer()
        self.vocab_size = self.tokenizer.vocab_size()
        self.transformer = Transformer(
            self.vocab_size,
            config.spectrogram.n_mels,
            config.model.n_encoder_layer,
            config.model.n_decoder_layer,
            config.model.n_head,
            config.model.n_hidden,
            config.model.d_model,
            config.model.dropout,
        )
        self.spectrogram = MelSpectrogram(
            config.spectrogram.sample_rate,
            config.spectrogram.n_fft,
            config.spectrogram.n_mels,
            config.spectrogram.hop_length,
        )
        self.loss_fn = CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)
        self.f1_score = MulticlassF1Score(
            self.vocab_size,
            average="weighted",
            multidim_average="global",
            ignore_index=self.tokenizer.pad_id,
            validate_args=False,
        )
        self.tgt_mask = self.transformer.get_subsequent_mask(config.dataset.tgt_seq_len)
        self.pad_id = self.tokenizer.pad_id
        self.config = config

    def forward(self, source, target):
        source = source.to(self.device)
        target = target.to(self.device)

        source = self.spectrogram.forward(source)
        tgt_mask = self.transformer.get_subsequent_mask(target.shape[1]).to(self.device)
        tgt_key_padding_mask = self.transformer.get_padding_mask(
            target, self.tokenizer.pad_id
        ).to(self.device)

        logits = self.transformer.forward(
            source, target, tgt_mask, tgt_key_padding_mask
        )
        return logits

    def training_step(self, batch, batch_idx):
        target = batch["tokens"][:, :-1]
        labels = batch["tokens"][:, 1:]

        frames = batch["frames"]
        frames = torch.flatten(frames, start_dim=1)
        source = self.spectrogram.forward(frames)

        tgt_mask = self.tgt_mask.to(self.device)
        tgt_key_padding_mask = Transformer.get_padding_mask(target, self.pad_id).to(
            self.device
        )

        logits = self.transformer.forward(
            source, target, tgt_mask, tgt_key_padding_mask
        )
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        target = batch["tokens"][:, :-1]
        labels = batch["tokens"][:, 1:]

        frames = batch["frames"]
        frames = torch.flatten(frames, start_dim=1)
        source = self.spectrogram.forward(frames)

        tgt_mask = self.tgt_mask.to(target.device)
        tgt_key_padding_mask = Transformer.get_padding_mask(target, self.pad_id).to(
            target.device
        )

        logits = self.transformer.forward(
            source, target, tgt_mask, tgt_key_padding_mask
        )
        predictions = torch.argmax(logits, dim=1)

        f1_score = self.f1_score(predictions, labels)
        self.log("f1_score", f1_score)

    def configure_optimizers(self):
        optimizer = Adafactor(
            self.transformer.parameters(),
            self.config.train.lr,
            scale_parameter=False,
            relative_step=False,
        )
        schedule = {
            "scheduler": get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.config.train.warmup_steps,
                num_training_steps=self.config.train.total_steps,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [schedule]
