from typing import Tuple
import PIL
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from PIL.Image import Image as PILImage
from augmentations import get_augmentations
import numpy as np
import pandas as pd

path = Path("../data/")


def list_files(path: Path):
    return [o for o in path.iterdir()]


class CassavaDataset(Dataset):
    def __init__(self, path, df, transform=None) -> None:
        super().__init__()
        self.df = df
        self.path = path
        self.transform = transform

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


if __name__ == "__main__":
    # print(list_files(path))
    train_tfms, test_tfms = get_augmentations(p=0.5, image_size=224)
    df = pd.read_csv(path / "train.csv")
    ds = Cassava(path=path / "train_images/", df=df, transform=train_tfms)
    img, label = ds[0]
    print(len(ds))
