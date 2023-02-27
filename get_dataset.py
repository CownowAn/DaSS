import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


def get_dataset(data: str, train: bool, transform=None, 
                target_transform=None, default_data_path=None, image_size=None) -> Dataset:
    if data == "cub":
        return CUB(default_data_path, train, transform, target_transform)
    elif data == "dtd":
        return DTD(default_data_path, train, transform, target_transform)
    elif data == "quickdraw":
        return QuickDraw(default_data_path, train, transform, target_transform)
    elif data == "stanford_cars":
        return StanfordCars(default_data_path, train, transform, target_transform)
    elif data == "tiny_imagenet":
        return TinyImageNet(default_data_path, train, transform, target_transform)
    else:
        raise NotImplementedError()


class NumpyDataset(Dataset):
    def __init__(self, default_data_path, image_path, label_path, transform=None, target_transform=None):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.image_path = os.path.join(default_data_path, image_path)
        self.images = np.load(os.path.join(default_data_path, image_path))
        self.labels = np.load(os.path.join(default_data_path, label_path))
        self.length = len(self.labels)

    def __getitem__(self, index):
        img = Image.fromarray(self.images[index])
        if 'quickdraw' in self.image_path:
            img = img.convert('RGB')
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return self.length


class CUB(NumpyDataset):
    def __init__(self, default_data_path, train=True, transform=None, target_transform=None):
        super().__init__(
            default_data_path=default_data_path, 
            image_path="cub/{}_images.npy".format("train" if train else "test"),
            label_path="cub/{}_labels.npy".format("train" if train else "test"),
            transform=transform,
            target_transform=target_transform,
        )


class DTD(NumpyDataset):
    def __init__(self, default_data_path, train=True, transform=None, target_transform=None):
        super().__init__(
            default_data_path=default_data_path, 
            image_path="dtd/{}_images.npy".format("train" if train else "test"),
            label_path="dtd/{}_labels.npy".format("train" if train else "test"),
            transform=transform,
            target_transform=target_transform,
        )


class QuickDraw(NumpyDataset):
    def __init__(self, default_data_path, train=True, transform=None, target_transform=None):
        super().__init__(
            default_data_path=default_data_path, 
            image_path="quickdraw/{}_images.npy".format("train" if train else "test"),
            label_path="quickdraw/{}_labels.npy".format("train" if train else "test"),
            transform=transform,
            target_transform=target_transform,
        )


class StanfordCars(NumpyDataset):
    def __init__(self, default_data_path, train=True, transform=None, target_transform=None):
        super().__init__(
            default_data_path=default_data_path, 
            image_path="stanford_cars/{}_images.npy".format("train" if train else "test"),
            label_path="stanford_cars/{}_labels.npy".format("train" if train else "test"),
            transform=transform,
            target_transform=target_transform,
        )


class TinyImageNet(NumpyDataset):
    def __init__(self, default_data_path, train=True, transform=None, target_transform=None):
        super().__init__(
            default_data_path=default_data_path, 
            image_path="tiny_imagenet/{}_images.npy".format("train" if train else "valid"),
            label_path="tiny_imagenet/{}_labels.npy".format("train" if train else "valid"),
            transform=transform,
            target_transform=target_transform,
        )
