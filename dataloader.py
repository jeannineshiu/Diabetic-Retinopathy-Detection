import pandas as pd
from torch.utils import data
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.transforms import transforms
import os
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
from imblearn.over_sampling import SMOTE


def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        self.data_size = len(self.img_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)
    
    def get_labels(self):
        return self.label
    
    def transform(self, img):
        # train mode
        if self.mode == 'train':
            # Center crop
            if img.size[1]<img.size[0]:
                crop_size = img.size[1]
            else:
                crop_size = img.size[0]
            center_img = transforms.CenterCrop(crop_size)(img)

            # resize to 512x512
            resize = transforms.Resize([512,512])
            resize_img = resize(center_img)
            img = resize_img

            # Random horizontal flipping
            if random.random() > 0.5:
                img = TF.hflip(img)

            # Random rotate
            rotate_angle = random.choice([0, 90, 180, 270])
            img = TF.rotate(img, rotate_angle)

            # converts its format from (N, H, W, C) to (N,C,H,W)
            # first to tensor then normalize
            img = TF.to_tensor(img)
            img = TF.normalize(img, [0.3, 0.2, 0.2], [0.2, 0.15, 0.15])

            return img
    
        else:
            # Center crop
            if img.size[1]<img.size[0]:
                crop_size = img.size[1]
            else:
                crop_size = img.size[0]
            center_img = transforms.CenterCrop(crop_size)(img)
            # resize to 512x512
            resize = transforms.Resize([512,512])
            resize_img = resize(center_img)
            img = resize_img

            img = TF.to_tensor(img)
            img = TF.normalize(img, [0.3, 0.2, 0.2], [0.2, 0.15, 0.15])

            return img

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        # load image from path
        img = Image.open(os.path.join(self.root, self.img_name[index] + '.jpeg')).convert('RGB')

        trans_img = self.transform(img)
   
        label = self.label[index]
   
        return trans_img, label

