from __future__ import annotations
from warnings import filterwarnings

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from model import OsuTransformer
from config.config import Config
from data.datamodule import OszDataModule

config = Config()
datamodule = OszDataModule(config)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    pl.seed_everything(config.seed)

    if config.train.resume:
        model = OsuTransformer.load_from_checkpoint(
            checkpoint_path=config.train.resume_checkpoint_path,
            config=config,
        )
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
        max_steps=config.train.session_steps,
        val_check_interval=config.val.interval,
        limit_val_batches=config.val.batches,
    )

    trainer.fit(model, datamodule=datamodule)
