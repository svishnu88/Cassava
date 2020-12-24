from geffnet import config
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from torch.optim import optimizer
from models import Resnext, get_efficientnet
from pathlib import Path
import pandas as pd
from cassavadata import *
import torch.nn.functional as F
import torch
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from losses import FocalLoss, LabelSmoothingCrossEntropy
from torch import optim
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import geffnet
from torch.optim.lr_scheduler import OneCycleLR
import math
from opt_utils import add_weight_decay
from warmup_scheduler.scheduler import GradualWarmupScheduler
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import os


ssl_models = [
    "resnet18_ssl",
    "resnet50_ssl",
    "resnext50_32x4d_ssl",
    "resnext101_32x4d_ssl",
    "resnext101_32x8d_ssl",
    "resnext101_32x16d_ssl",
]

eff_models = ["tf_efficientnet_b3_ns,tf_efficientnet_b4_ns"]

loss_fns = {
    "cross_entropy": F.cross_entropy,
    "focal_loss": FocalLoss(),
    "label_smoothing": LabelSmoothingCrossEntropy(smoothing=0.3),
}


class CassavaModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = None,
        num_classes: int = None,
        loss_fn: str = "cross_entropy",
        lr=1e-4,
        wd=1e-6,
    ):
        super().__init__()

        if model_name.find("res") > -1:
            self.model = Resnext(model_name=model_name, num_classes=num_classes)
        elif model_name.find("effi") > -1:
            self.model = get_efficientnet(model_name)
        self.loss_fn = loss_fns[loss_fn]
        self.lr = lr
        self.accuracy = pl.metrics.Accuracy()
        self.wd = wd

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("valid_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy(y_hat, y), prog_bar=True)

    def configure_optimizers(self):
        # parameters = add_weight_decay(self.model, self.wd)
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
        )

        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.max_epochs, 0
        )
        warmup_scheduler = GradualWarmupScheduler(
            optimizer, multiplier=4, total_epoch=2, after_scheduler=cosine_scheduler,
        )

        return [optimizer], [warmup_scheduler]


@hydra.main(config_path="conf", config_name="config")
def cli_hydra(cfg: DictConfig):
    pl.seed_everything(1234)

    wandb_logger = instantiate(cfg.wandb)
    wandb_logger.log_hyperparams(cfg)

    data_module = instantiate(cfg.data)
    data_module.prepare_data()
    data_module.setup()
    model = instantiate(cfg.model)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    weights_path = Path(f"weights/")
    checkpoint_callback = ModelCheckpoint(
        dirpath=weights_path,
        save_weights_only=True,
        monitor="val_acc",
        mode="max",
        filename=f"{cfg.data.fold_id}",
    )

    trainer = pl.Trainer(
        # accelerator="ddp",
        callbacks=[lr_monitor, checkpoint_callback],
        logger=[wandb_logger],
        gpus=cfg.trainer.gpus,
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=0.1,
        precision=cfg.trainer.precision,
        sync_batchnorm=cfg.trainer.sync_bn,
    )

    trainer.fit(model=model, datamodule=data_module)


def cli_main():

    # ------------
    # args
    # ------------

    parser = CassavaModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # Log Metrics using Wandb
    # ------------

    wandb_logger = WandbLogger(name="Initial-Pipeline", project="Cassava")

    # ------------
    # Create Data Module
    # ------------

    data_module = CassavaDataModule(
        path=args.path,
        aug_p=args.aug_p,
        val_pct=args.val_pct,
        img_sz=args.img_sz,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fold_id=args.fold_id,
    )
    data_module.prepare_data()
    data_module.setup()

    # ------------
    # Create Model
    # ------------

    model = CassavaModel(
        model_name=args.model_name,
        num_classes=args.num_classes,
        data_path=args.data_path,
        lr=args.lr,
        loss_fn=loss_fn[args.loss_fn],
    )

    # ------------
    # Training
    # ------------

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    weights_path = Path(f"weights/")
    checkpoint_callback = ModelCheckpoint(
        dirpath=weights_path,
        save_weights_only=True,
        monitor="val_acc",
        mode="max",
        filename=f"{args.fold_id}",
    )
    trainer = pl.Trainer(
        accelerator="ddp",
        callbacks=[lr_monitor, checkpoint_callback],
        logger=[wandb_logger],
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        gradient_clip_val=0.1,
        precision=args.precision,
        sync_batchnorm=True,
    )

    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    cli_hydra()
