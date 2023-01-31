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
from sklearn.preprocessing import LabelEncoder

eval_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
color2label = {
    (128,64,128) : 0,
    (244,35,232) : 1,
    (70,70,70) : 2,
    (102,102,156): 3,
    (190,153,153) : 4,
    (153,153,153) : 5,
    (250,170,30): 6,
    (220,220,0) : 7,
    (107,142,35) : 8,
    (152,251,152) : 9,
    (70,130,180) : 10,
    (220,20,60) : 11,
    (255,0,0) : 12,
    (0,0,142) : 13,
    (0,0,70) : 14,
    (0,60,100) : 15,
    (0,80,100) : 16,
    (0,0,230) : 17,
    (119,11,32) : 18 ,
    (0,0,0): 19
}

def rgb2index(rgb):
  if rgb in list(color2label.keys()):
    return color2label[tuple(rgb)]
  else:
    return 19

def mask_to_class_rgb( mask):
        #print('----mask->rgb----')
        mask = torch.from_numpy(np.array(mask))
        mask = torch.squeeze(mask)  # remove 1

        # check the present values in the mask, 0 and 255 in my case
        print('unique values rgb    ', torch.unique(mask)) 
        # -> unique values rgb     tensor([  0, 255], dtype=torch.uint8)

        class_mask = mask
        class_mask = class_mask.permute(2, 0, 1).contiguous()
        h, w = class_mask.shape[1], class_mask.shape[2]
        mask_out = torch.empty(h, w, dtype=torch.long)

        for k in color2label:
            idx = (class_mask == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))         
            validx = (idx.sum(0) == 3)          
            mask_out[validx] = torch.tensor(color2label[k], dtype=torch.long)

        # check the present values after mapping, in my case 0, 1, 2, 3
        print('unique values mapped ', torch.unique(mask_out))
        # -> unique values mapped  tensor([0, 1, 2, 3])
       
        return mask_out

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
          print(label)
          #np.apply_along_axis(rgb2index, 0, label)
          label = mask_to_class_rgb(label)
          #label = label[0].squeeze()




        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.data) # Provide a way to get the length (number of elements) of the dataset
        return length