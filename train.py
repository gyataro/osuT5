import torch
import pytorch_lightning as pl
from torch.nn import NLLLoss
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from config.config import Config
from tokenizer import Tokenizer
from model import Transformer
from spectrogram import MelSpectrogram
from optimizer import Adafactor, get_constant_schedule_with_warmup
from data.datamodule import OszDataModule


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
                last_epoch=-1,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [schedule]


if __name__ == "__main__":
    config = Config()
    pl.seed_everything(config.seed)
    model = OsuTransformer(config)
    datamodule = OszDataModule(config)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_top_k=3,
        save_weights_only=False,
    )

    trainer = pl.Trainer(
        precision=32,
        callbacks=[lr_monitor, model_checkpoint],
        accelerator="auto",
        max_steps=config.train.num_steps,
        val_check_interval=10,
    )
    trainer.fit(model, datamodule=datamodule)
