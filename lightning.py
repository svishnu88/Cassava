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
        self, model_name: str = None, num_classes: int = None, data_path: Path = None
    ):
        super().__init__()
        self.model = Resnext(model_name=model_name, num_classes=num_classes)
        self.data_path = data_path

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("valid_loss", loss, on_step=True)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        return parser

    def cli_main():
        pl.seed_everything(1234)

        # ------------
        # args
        # ------------
        parser = ArgumentParser()
        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--hidden_dim", type=int, default=128)
        parser = pl.Trainer.add_argparse_args(parser)
        parser = CassavaModel.add_model_specific_args(parser)
        args = parser.parse_args()
