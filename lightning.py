from geffnet import config
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from models import Resnext, get_efficientnet
from pathlib import Path
import pandas as pd
from cassavadata import *
import torch.nn.functional as F
import torch
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from losses import FocalLoss
from torch import optim
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import geffnet


ssl_models = [
    "resnet18_ssl",
    "resnet50_ssl",
    "resnext50_32x4d_ssl",
    "resnext101_32x4d_ssl",
    "resnext101_32x8d_ssl",
    "resnext101_32x16d_ssl",
]

eff_models = ["tf_efficientnet_b3_ns,tf_efficientnet_b4_ns"]

loss_fn = {"cross_entropy": F.cross_entropy, "focal_loss": FocalLoss()}


class CassavaModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = None,
        num_classes: int = None,
        data_path: Path = None,
        loss_fn=F.cross_entropy,
        lr=1e-4,
        wd=1e-6,
    ):
        super().__init__()

        if model_name.find("res") > -1:
            self.model = Resnext(model_name=model_name, num_classes=num_classes)
        elif model_name.find("effi") > -1:
            self.model = get_efficientnet(model_name)
        self.data_path = data_path
        self.loss_fn = loss_fn
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
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.max_epochs, 0
        )

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--model_name", default=ssl_models[2], type=str)
        parser.add_argument("--num_classes", default=5, type=int)
        parser.add_argument("--data_path", default="../data/", type=str)
        parser.add_argument("--lr", default=0.0001, type=float)
        parser.add_argument("--wd", default=1e-6, type=float)
        parser.add_argument("--loss_fn", default="cross_entropy", type=str)

        return parser


def cli_main():
    pl.seed_everything(1234)
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--aug_p", type=float, default=0.5)
    parser.add_argument("--val_pct", type=float, default=0.2)
    parser.add_argument("--img_sz", type=int, default=224)
    parser.add_argument("--path", type=str, default="../data/")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--gpus", type=int, default=-1)
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--fold_id", type=int, default=0)
    parser.add_argument("--precision", type=int, default=16)

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

    lr_monitor = LearningRateMonitor(logging_interval="step")
    weights_path = Path(f"weights/")
    checkpoint_callback = ModelCheckpoint(
        dirpath=weights_path,
        save_weights_only=True,
        monitor="val_acc",
        mode="max",
        save_last=True,
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
    cli_main()
