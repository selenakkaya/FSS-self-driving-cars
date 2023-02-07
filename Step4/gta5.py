from torchvision.datasets import VisionDataset

from PIL import Image

import os
import random
import pandas as pd
import numpy as np
from torch import from_numpy
import matplotlib.pyplot as plt
from torchvision import transforms
import torch

palette = {
    (128, 64, 128): 0,
    (244, 35, 232): 1,
    (70, 70, 70): 2,
    (102, 102, 156): 3,
    (190, 153, 153): 4,
    (153, 153, 153): 5,
    (250, 170, 30): 6,
    (220, 220, 0): 7,
    (107, 142, 35): 8,
    (152, 251, 152): 9,
    (70, 130, 180): 10,
    (220, 20, 60): 11,
    (255, 0, 0): 12,
    (0, 0, 142): 13,
    (0, 0, 70): 14,
    (0, 60, 100): 15,
    (0, 80, 100): 16,
    (0, 0, 230): 17,
    (119, 11, 32): 18,
    (0, 0, 0): 255
}

class GTA5(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(GTA5, self).__init__(root, transform=transform,
                                   target_transform=target_transform)
        self.path = root + 'data/GTA5/'
        self.palette = palette
        self.styles = None
        self.return_PIL = False

        images_path = self.path + 'images'
        labels_path = self.path + 'labels'

        images = []
        labels = []

        for line in open(f"{root}data/GTA5/train.txt"):
            line = line.replace("\n", "")
            i = line
            l = line
            images.append(i)
            labels.append(l)

        # print(images)
        self.data = pd.DataFrame(zip(images, labels), columns=["image", "label"])

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''
        image_path = self.path + 'images/' + self.data["image"].iloc[index]
        label_path = self.path + 'labels/' + self.data["label"].iloc[index]
        image_PIL = Image.open(image_path)
        image, label = plt.imread(image_path), plt.imread(label_path)*255

        # Applies preprocessing when accessing the image
        if self.styles is not None:
            image_PIL = self.styles.apply_style(image_PIL)
            image = image_PIL
        if self.return_PIL:
            return image

        if self.transform is not None:
            image = transforms.ToTensor()(image)
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.rgb2class_index(label)
            label = transforms.ToTensor()(label)
            label = label.type(torch.LongTensor)
            label = self.target_transform(label)

        return image, label

    def set_styles(self, styles):
        self.styles = styles

    def rgb2class_index(self, img):
        k = np.array(list(self.palette.keys()))
        v = np.array(list(self.palette.values()))

        s = 256**np.arange(3)
        k1D = k.dot(s)

        sidx = k1D.argsort()
        k1Ds = k1D[sidx]
        vs = v[sidx]

        labelOld2D = np.tensordot(img, s, axes=((-1),(-1)))

        out = vs[np.searchsorted(k1Ds, labelOld2D)]
        return out

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.data)
        return length
