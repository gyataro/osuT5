from __future__ import annotations
from warnings import filterwarnings

import torch
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss
from torchmetrics import CosineSimilarity
from torchmetrics.classification import MulticlassF1Score
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from config.config import Config
from module.transformer import Transformer
from module.spectrogram import MelSpectrogram
from optimization.optimizer import Adafactor
from optimization.scheduler import get_constant_schedule_with_warmup
from data.tokenizer import Tokenizer
from data.datamodule import OszDataModule

filterwarnings("ignore", ".*does not have many workers.*")


class OsuTransformer(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
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
        self.loss_fn = CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id, label_smoothing=0.1
        )
        self.f1_score = MulticlassF1Score(
            self.vocab_size,
            average="weighted",
            multidim_average="global",
            ignore_index=self.tokenizer.pad_id,
            validate_args=False,
        )
        self.cosine_similarity = CosineSimilarity(reduction="mean")
        self.tgt_mask = self.transformer.get_subsequent_mask(config.dataset.tgt_seq_len)
        self.pad_id = self.tokenizer.pad_id
        self.config = config

    def training_step(self, batch, batch_idx):
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
        cosine_similarity = self.cosine_similarity(predictions, labels)
        self.log("fl_score", f1_score)
        self.log("cosine_similarity", cosine_similarity)

    def configure_optimizers(self):
        optimizer = Adafactor(
            self.transformer.parameters(),
            config.train.lr,
            scale_parameter=False,
            relative_step=False,
        )
        schedule = {
            "scheduler": get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=config.train.warmup_steps,
                last_epoch=self.current_epoch - 1,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [schedule]


config = Config()
datamodule = OszDataModule(config)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    pl.seed_everything(config.seed)

    if config.train.resume:
        model = OsuTransformer.load_from_checkpoint(config.train.resume_checkpoint_path)
    else:
        model = OsuTransformer(config)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(
        monitor="f1_score",
        mode="max",
        save_last=True,
        save_top_k=3,
        save_weights_only=False,
        filename="{step}-{f1_score:.2f}",
    )

    trainer = pl.Trainer(
        precision=32,
        callbacks=[lr_monitor, model_checkpoint],
        accelerator="auto",
        max_steps=config.train.num_steps,
        val_check_interval=config.val.interval,
        limit_val_batches=config.val.batches,
    )

    trainer.fit(model, datamodule=datamodule)
