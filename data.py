from typing import Tuple
import PIL
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from PIL.Image import Image as PILImage
from torch.utils.data.dataloader import DataLoader
from augmentations import get_augmentations
import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
import os

path = Path("../data/")


def list_files(path: Path):
    return [o for o in path.iterdir()]


class CassavaDataset(Dataset):
    def __init__(self, path, df, transform=None) -> None:
        super().__init__()
        self.df = df
        self.path = path
        self.transform = transform
        self.num_workers = 2

    def __getitem__(self, index) -> Tuple[PILImage, int]:
        img_id, label = self.df.iloc[index]
        image = Image.open(self.path / img_id)
        image = np.array(image)
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label

    def __len__(self):
        return self.df.shape[0]


class CassavaDataModule(LightningDataModule):
    def __init__(
        self,
        path: str = None,
        aug_p: float = 0.5,
        val_pct: float = 0.2,
        img_sz: int = 224,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.path = Path(path)
        self.aug_p = aug_p
        self.val_pct = val_pct
        self.img_sz = img_sz
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # only called on 1 GPU/TPU in distributed
        df = pd.read_csv(self.path / "train.csv")
        train_df, valid_df = train_test_split(
            df, test_size=self.val_pct, random_state=42, stratify=df.label
        )
        train_df.to_pickle(self.path / "train_df.pkl")
        valid_df.to_pickle(self.path / "valid_df.pkl")

    def setup(self):
        # called on every process in DDP
        self.train_transform, self.test_transform = get_augmentations(
            p=self.aug_p, image_size=self.img_sz
        )
        self.train_df = pd.read_pickle(self.path / "train_df.pkl")
        self.valid_df = pd.read_pickle(self.path / "valid_df.pkl")

    def train_dataloader(self):
        train_dataset = CassavaDataset(
            self.path / "train_images", df=self.train_df, transform=self.train_transform
        )
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        valid_dataset = CassavaDataset(
            self.path / "train_images", df=self.valid_df, transform=self.test_transform
        )
        return DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


if __name__ == "__main__":
    # Test Cassava Test Module
    path = "../data"
    data = CassavaDataModule(
        path=path, aug_p=0.5, val_pct=0.2, img_sz=224, batch_size=64
    )
    data.prepare_data()
    data.setup()
    xb, yb = next(iter(data.train_dataloader()))
    print(xb.shape)
    print(yb.shape)
