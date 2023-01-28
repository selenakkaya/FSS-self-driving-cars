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

eval_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

class GTA5(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(GTA5, self).__init__(root, transform=transform, target_transform=target_transform)
        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        '''


        
        self.path = root + 'data/GTA5/'
        images_path = self.path + 'images'
        labels_path = self.path + 'labels'

        images = []
        labels = []
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for i, cl in enumerate(eval_classes):
          mapping[cl] = i
        self.label_remapping = lambda x: from_numpy(mapping[x])
        
        for line in open(f"{root}data/GTA5/train.txt"):
                line = line.replace("\n", "")
                i = line
                l = line
                images.append(i)
                labels.append(l)
        
        
        #print(images)
        self.data = pd.DataFrame(zip(images, labels), columns = ["image", "label"])

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
        #image, label = Image.open(image_path), Image.open(label_path)
        image, label = plt.imread(image_path), plt.imread(label_path)*255
        
        # Applies preprocessing when accessing the image
        if self.transform is not None:
          image = transforms.ToTensor()(image)
          image = self.transform(image)
        if self.target_transform is not None:
          label = transforms.ToTensor()(label)
          label = label.type(torch.LongTensor)
          label = self.target_transform(label)
          label = self.label_remapping(label)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.data) # Provide a way to get the length (number of elements) of the dataset
        return length
