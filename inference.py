import pandas as pd
from data import CassavaDataset
from pathlib import Path
from augmentations import get_augmentations, get_tta
from torch.utils.data import DataLoader
from lightning import CassavaModel
import torch
from models import Resnext


test_df = pd.read_csv("../data/sample_submission.csv")
path = Path("../data/")
batch_size, num_workers = 32, 8
ssl_models = [
    "resnet18_ssl",
    "resnet50_ssl",
    "resnext50_32x4d_ssl",
    "resnext101_32x4d_ssl",
    "resnext101_32x8d_ssl",
    "resnext101_32x16d_ssl",
]

tta_tfms = get_tta(image_size=512)
test_ds = CassavaDataset(path=path / "test_images", df=test_df, transform=tta_tfms)

test_dl = DataLoader(
    dataset=test_ds,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=True,
)

device = torch.device("cuda")
model = Resnext(model_name=ssl_models[2], num_classes=5, kaggle=True)
model = model.to(device)

chk_path = "Cassava/jim2mgsp/checkpoints/epoch=4-step=224.ckpt"
chk = torch.load(chk_path)
model_weights = {k.replace("model.", ""): v for k, v in chk["state_dict"].items()}
torch.save(model_weights, "model_weights.pth")

model.load_state_dict(model_weights)

preds = []
with torch.no_grad():
    for xb, _ in test_dl:
        xb = xb.to(device)
        pred = model(xb)
        preds.extend(pred.argmax(1).to("cpu").tolist())

test_df.label = preds
test_df.to_csv("submission.csv", index=False)
