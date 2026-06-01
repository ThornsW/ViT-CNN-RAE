"""Fine-tune torchvision classifiers on Caltech-256.

A single train_classifier(name, ...) replaces the three near-duplicate scripts
that the upstream repo shipped (densenet121 / resnet50 / mobilenet_v3_large).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models as tv_models
from torchvision.models import DenseNet121_Weights, MobileNet_V3_Large_Weights, ResNet50_Weights

from .. import config
from ..data import MyDataset, default_transform


_TRAIN_BUILDERS = {
    'densenet121': {
        'ctor': lambda: tv_models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1),
        'replace': lambda m, n: setattr(m, 'classifier', nn.Linear(m.classifier.in_features, n)),
        'default_ckpt': 'DenseNet121.pth',
    },
    'resnet50': {
        'ctor': lambda: tv_models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
        'replace': lambda m, n: setattr(m, 'fc', nn.Linear(m.fc.in_features, n)),
        'default_ckpt': 'ResNet50.pth',
    },
    'mobilenet_v3_large': {
        'ctor': lambda: tv_models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1),
        'replace': lambda m, n: m.classifier.__setitem__(
            3, nn.Linear(m.classifier[3].in_features, n)),
        'default_ckpt': 'MoblieNetV3.pth',
    },
}


def train_classifier(name: str = 'densenet121',
                     epochs: int = 30,
                     batch_size: int = 128,
                     num_classes: int = config.NUM_CLASSES,
                     out_path: Path | str | None = None) -> Path:
    """Fine-tune a torchvision classifier on Caltech-256 and save best checkpoint.

    Returns the path to the saved .pth.
    """
    if name not in _TRAIN_BUILDERS:
        raise ValueError(f"unknown classifier: {name}. Choose from {list(_TRAIN_BUILDERS)}.")
    spec = _TRAIN_BUILDERS[name]
    out_path = Path(out_path) if out_path else config.checkpoint_path(spec['default_ckpt'])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_classifier] {name} on {device} → {out_path}")

    train_data = MyDataset(txt=config.split_path('dataset-trn.txt'),
                           root=config.DATA_ROOT,
                           transform=default_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              pin_memory=torch.cuda.is_available(), num_workers=4)

    test_data = MyDataset(txt=config.split_path('dataset-val.txt'),
                          root=config.DATA_ROOT,
                          transform=default_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                             pin_memory=torch.cuda.is_available(), num_workers=4)

    model = spec['ctor']()
    for param in model.parameters():
        param.requires_grad = False
    spec['replace'](model, num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
    best_accuracy = 0.0

    for epoch in range(epochs):
        if epoch == 10:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        if epoch == 20:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

        model.train()
        loss_epoch = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss_epoch += float(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"loss in epoch {epoch}: {loss_epoch:.4f}")

        model.eval()
        num_correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                pred = torch.argmax(model(images), 1)
                num_correct += int(torch.sum(pred == labels))

        acc = num_correct / len(test_data)
        if acc > best_accuracy:
            best_accuracy = acc
            torch.save(model.state_dict(), out_path)
            print(f"epoch {epoch}, best accuracy {best_accuracy:.4f}")

    return out_path
