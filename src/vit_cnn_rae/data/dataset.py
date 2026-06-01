"""Caltech-256 dataset reader and the project's default image transform."""
from __future__ import annotations

import os

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    """Read (relative_path, label) pairs from a txt file and prefix them with `root`."""

    def __init__(self, txt, root='', transform=None, target_transform=None, loader=default_loader):
        with open(txt, 'r') as fh:
            imgs = []
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                words = line.split()
                imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.root = str(root)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        full_path = os.path.join(self.root, fn.lstrip(os.sep)) if self.root else fn
        img = self.loader(full_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


default_transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
