from torchvision.datasets import VisionDataset
from PIL import Image

import os
import random
import numpy as np
from torch import from_numpy
import matplotlib.pyplot as plt
from torchvision import transforms
import torch

eval_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

def get_label(image):
  spl = image.split("_")
  l = spl[0] + "_" + spl[1] + "_" + spl[2] + "_gtFine_labelIds.png"
  return l

class Cityscapes(VisionDataset):
    def __init__(self, root, partition_type, split='train', transform=None, target_transform=None):
        super(Cityscapes, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split
        self.client_index = 0

        self.path = root + 'data/Cityscapes/'
        images_path = self.path + 'images'
        labels_path = self.path + 'labels'

        images = []
        clients_images = [] #clients_images[i] corresponds to the list of images given to the i-th client

        mapping = np.zeros((256,), dtype=np.int64) + 255
        for i, cl in enumerate(eval_classes):
          mapping[cl] = i
        self.label_remapping = lambda x: from_numpy(mapping[x])

        #Partition A: 2 random images from every city for evaluation
        if partition_type[0] == 'A':
            cities, city_images_dict = self.get_city_images_dict(images_path)

            #selects 2 random images from each city
            random.seed(24)
            for city in cities:
                random_images = random.sample([*city_images_dict[city]], k=2)
                for r_i in random_images:
                    city_images_dict[city].remove(r_i)
                if split == 'train':
                    images.extend(city_images_dict[city])
                else: # split == 'val'
                    images.extend(random_images)

            if split == 'train':
              if partition_type[1] == 'heterogeneous':
                for c in cities:
                  l = len(city_images_dict[c])
                  if l > 20:
                    division = np.array_split(list(city_images_dict[c]), l/20+1)
                    for imglist in division:
                      clients_images.append(imglist)
                  else:
                    clients_images.append(list(city_images_dict[c]))
              elif partition_type[1] == 'uniform':
                random.shuffle(images)
                division = np.array_split(images, len(images)/20+1)
                for imglist in division:
                  clients_images.append(imglist)
              else:
                raise Exception(f'Invalid argument: {partition_type[1]}')

        #Partition B: use the split in train.txt and val.txt
        elif partition_type[0] == 'B':
            for line in open(f"{root}data/Cityscapes/{self.split}.txt"):
                line = line.replace("\n", "")
                i = line.split("/")[1]
                images.append(i)
            if split == 'val':
                pass #all is ok
            else:    #split == 'train'
                if partition_type[1] == 'heterogeneous':
                    cities, city_images_dict = self.get_city_images_dict(images_path)
                    for c in cities:
                        for img in list(city_images_dict[c]):
                            if img not in images:
                                city_images_dict[c].remove(img)
                        l = len(city_images_dict[c])
                        if l > 20:
                            division = np.array_split(list(city_images_dict[c]), l/20+1)
                            for imglist in division:
                                clients_images.append(imglist)
                        else:
                            if city_images_dict[c]:
                                clients_images.append(list(city_images_dict[c]))
                elif partition_type[1] == 'uniform':
                    random.shuffle(images)
                    division = np.array_split(images, len(images)/20+1)
                    for imglist in division:
                        clients_images.append(imglist)
                else:
                    raise Exception(f'Invalid argument: {partition_type[1]}')
        else:
            raise Exception(f'Invalid argument: {partition_type[0]}')
        
        if split == 'train':
            self.data = clients_images
        else: #split == 'val'
            self.data = images
        self.num_clients = len(clients_images)

    def set_client(self, index):
      '''Sets the client whose images will be returned by __getitem__'''
      if self.client_index > self.num_clients:
        self.client_index = self.num_clients - 1
      self.client_index = index

    def get_num_clients(self):
      return self.num_clients

    def get_city_images_dict(self, images_path):
      #creates a set of city names, used to create the city_image_dict
      cities = set()
      files = os.listdir(images_path)
      for f in files:
        city = f.split("_")[0]
        cities.add(city)

      #intializes the city_image_dict dictionary
      #key: city name
      #value: list of images from that city
      city_images_dict = dict()
      for city in cities:
        city_images_dict[city] = set()
      for f in files:
        city = f.split("_")[0]
        city_images_dict[city].add(f)

      return cities, city_images_dict

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''
        if self.split == 'train':
            image_name = self.data[self.client_index][index]
        else:
            image_name = self.data[index]
        image_path = self.path + 'images/' + image_name
        label_path = self.path + 'labels/' + get_label(image_name)
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
        if self.split == 'train':
            length = len(self.data[self.client_index]) # Provide a way to get the length (number of elements) of the dataset
        else: #split == 'val':
            length = len(self.data)
        return length
