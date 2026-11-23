"""Dataset / DataLoader 与预处理。

训练集使用随机增强，验证/测试集使用确定性预处理（缩放 + 中心裁剪）。
Web 推理复用 build_eval_transform()，保证训练与部署的输入分布一致。
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import config


def build_train_transform(img_size: int = config.IMG_SIZE, augment: bool = True):
    if not augment:
        # 对照组：只做缩放与标准化，不引入随机增强
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD),
        ])
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])


def build_eval_transform(img_size: int = config.IMG_SIZE):
    return transforms.Compose([
        transforms.Resize(config.RESIZE_SIZE),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])


def build_datasets(img_size: int = config.IMG_SIZE, augment: bool = True):
    train_ds = ImageFolder(config.TRAIN_DIR, transform=build_train_transform(img_size, augment))
    val_ds = ImageFolder(config.VAL_DIR, transform=build_eval_transform(img_size))
    test_ds = ImageFolder(config.TEST_DIR, transform=build_eval_transform(img_size))
    assert train_ds.classes == config.CLASS_NAMES, f"类别顺序不一致: {train_ds.classes}"
    return train_ds, val_ds, test_ds


def build_loaders(batch_size: int = config.BATCH_SIZE, num_workers: int = config.NUM_WORKERS,
                  img_size: int = config.IMG_SIZE, augment: bool = True):
    train_ds, val_ds, test_ds = build_datasets(img_size, augment)
    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


class SingleImageDataset(torch.utils.data.Dataset):
    """供 predict.py 使用：单张图片 -> 张量。"""

    def __init__(self, image_path, transform):
        from PIL import Image
        self.image = Image.open(image_path).convert("RGB")
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.transform(self.image)
