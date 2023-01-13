from torchvision.datasets import VisionDataset

from PIL import Image

import os
import random
import pandas as pd

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class Cityscapes(VisionDataset):
    def __init__(self, root, partition_type, split='train', transform=None, target_transform=None):
        super(Cityscapes, self).__init__(root, transform=transform, target_transform=target_transform)
        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        '''

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'val.txt')
        
        self.path = root + 'data/Cityscapes/'
        images_path = self.path + 'images'
        labels_path = self.path + 'labels'

        images = []
        labels = []
        
        #Partition A: 2 random images from every city for evaluation
        if partition_type == 'A':
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

            #selects 2 random images from each cities and puts them in images[]
            random.seed(24)
            for city in cities:
                random_images = random.sample([*city_images_dict[city]], k=2)
                for r_i in random_images:
                    city_images_dict[city].remove(r_i)
                if split == 'train':
                    images.extend(city_images_dict[city])
                else:
                    images.extend(random_images)

            #fetches the corresponding labels
            for i in images:
                spl = i.split("_")
                l = spl[0] + "_" + spl[1] + "_" + spl[2] + "_gtFine_labelIds.png"
                labels.append(l)
            
        #Partition B: use the split in train.txt and val.txt
        elif partition_type == 'B':
            for line in open(f"{root}data/Cityscapes/{self.split}.txt"):
                line = line.replace("\n", "")
                i = line.split("/")[1]
                spl = i.split("_")
                l = spl[0] + "_" + spl[1] + "_" + spl[2] + "_gtFine_labelIds.png"
                images.append(i)
                labels.append(l)
        else:
            print("Cityscapes::__init__ Invalid partiton_type argument")
        
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
        image, label = pil_loader(image_path), pil_loader(label_path)

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.data) # Provide a way to get the length (number of elements) of the dataset
        return length
