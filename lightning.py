import pytorch_lightning as pl
from models import Resnext
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from data import *
from augmentations import get_augmentations
from torch.utils.data import DataLoader
from pytorch_lightning import _logger as log

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

    def setup(self):
        df = pd.read_csv(self.data_path / "train.csv")
        train_df, valid_df = train_test_split(
            df, test_size=0.30, random_state=42, stratify=df.label
        )
        train_transform, test_transform = get_augmentations(p=0.5, image_size=224)
        train_dataset = CassavaDataset(
            self.data_path, df=train_df, transform=train_transform
        )
        valid_dataset = CassavaDataset(
            self.data_path, df=valid_df, transform=test_transform
        )
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def __dataloader(self, train):
        """Train/validation loaders."""
        _dataset = self.train_dataset if train else self.valid_dataset
        loader = DataLoader(
            dataset=_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True if train else False,
        )

        return loader

    def train_dataloader(self):
        log.info("Training data loaded.")
        return self.__dataloader(train=True)

    def val_dataloader(self):
        log.info("Validation data loaded.")
        return self.__dataloader(train=False)
