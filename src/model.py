"""模型定义：自定义 CNN、ResNet18、MobileNetV3-Small。

统一接口 build_model(name, num_classes, pretrained=True)。
使用 ImageNet 预训练权重时替换最后的全连接/分类层，其余层在本项目数据上重新训练。
"""
import torch
import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    """从零训练的小型卷积网络（作为基线实验，不使用任何预训练权重）。"""

    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # 112
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # 56
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # 28
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name: str, num_classes: int = 6, pretrained: bool = True) -> nn.Module:
    """构建模型。name 取值：smallcnn / resnet18 / mobilenet_v3。"""
    name = name.lower()

    if name == "smallcnn":
        return SmallCNN(num_classes)

    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes),
        )
        return model

    if name in {"mobilenet_v3", "mobilenetv3", "mobilenet_v3_small"}:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"未知模型名称: {name}")


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def set_backbone_frozen(model: nn.Module, name: str, frozen: bool = True) -> None:
    """冻结/解冻主干网络，只训练分类头（用于迁移学习的 warm-up 阶段）。"""
    name = name.lower()
    if name == "resnet18":
        for p in model.parameters():
            p.requires_grad = True
        if frozen:
            for p in model.conv1.parameters():
                p.requires_grad = False
            for p in model.bn1.parameters():
                p.requires_grad = False
            for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
                for p in layer.parameters():
                    p.requires_grad = False
    elif name.startswith("mobilenet"):
        for p in model.features.parameters():
            p.requires_grad = not frozen
        for p in model.classifier.parameters():
            p.requires_grad = True


if __name__ == "__main__":
    for n in ("smallcnn", "resnet18", "mobilenet_v3"):
        m = build_model(n, 6, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        y = m(x)
        print(f"{n:14s} 参数量={count_parameters(m):,}  输出={tuple(y.shape)}")
