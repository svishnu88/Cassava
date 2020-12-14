from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
from hubconf_pretrained_false import (
    resnet18_ssl,
    resnet50_ssl,
    resnext50_32x4d_ssl,
    resnext101_32x4d_ssl,
    resnext101_32x8d_ssl,
    resnext101_32x16d_ssl,
)

ssl_models = [
    "resnet18_ssl",
    "resnet50_ssl",
    "resnext50_32x4d_ssl",
    "resnext101_32x4d_ssl",
    "resnext101_32x8d_ssl",
    "resnext101_32x16d_ssl",
]


class Resnext(nn.Module):
    def __init__(
        self,
        model_name="resnet18_ssl",
        pool_type=F.adaptive_avg_pool2d,
        num_classes=1000,
        kaggle=False,
    ):
        super().__init__()
        self.pool_type = pool_type

        if kaggle:
            backbone = eval(model_name)()
        else:
            backbone = torch.hub.load(
                "facebookresearch/semi-supervised-ImageNet1K-models", model_name
            )
        list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        in_features = getattr(backbone, "fc").in_features
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.pool_type(self.backbone(x), 1)
        features = features.view(x.size(0), -1)
        return self.classifier(features)


if __name__ == "__main__":
    model = Resnext(num_classes=5)
    sample_input = torch.rand(size=(5, 3, 224, 224))
    sample_targets = torch.randint(0, 5, size=(5,))

    output = model(sample_input)
    print(F.cross_entropy(output, sample_targets))
