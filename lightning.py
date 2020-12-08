import pytorch_lightning as pl
from models import Resnext
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from data import *
from augmentations import get_augmentations
from torch.utils.data import DataLoader
from pytorch_lightning import _logger as log
import torch.nn.functional as F
import torch
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger

ssl_models = [
    "resnet18_ssl",
    "resnet50_ssl",
    "resnext50_32x4d_ssl",
    "resnext101_32x4d_ssl",
    "resnext101_32x8d_ssl",
    "resnext101_32x16d_ssl",
]


class CassavaModel(pl.LightningModule):
    def __init__(
        self,
        model_num: int = 2,
        num_classes: int = None,
        data_path: Path = None,
        loss_fn=F.cross_entropy,
        lr=1e-4,
    ):
        super().__init__()
        self.model = Resnext(model_name=ssl_models[model_num], num_classes=num_classes)
        self.data_path = data_path
        self.loss_fn = loss_fn
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("valid_loss", loss, on_step=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--model_num", default=2, type=int)
        parser.add_argument("--num_classes", default=5, type=int)
        parser.add_argument("--data_path", default="../data/", type=str)
        parser.add_argument("--lr", default=0.0001, type=float)
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
    parser.add_argument("--gpus", "--gpus", type=int, default=1)

    # parser = pl.Trainer.add_argparse_args(parser)
    parser = CassavaModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # Log Metrics using Wandb
    # ------------

    wandb_logger = WandbLogger(name="Initial-Pipeline", project="Cassava Leaf Disease")
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
    )
    data_module.prepare_data()
    data_module.setup()

    # ------------
    # Create Model
    # ------------

    model = CassavaModel(
        model_num=args.model_num,
        num_classes=args.num_classes,
        data_path=args.data_path,
        lr=args.lr,
    )

    # ------------
    # Training
    # ------------

    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger,)
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    cli_main()
