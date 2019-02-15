import os
from os.path import isdir, exists, abspath, join
import torch
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, root_dir='inpainting_set', batch_size=16):
        self.mode = 'train'
        self.batch_size = batch_size
        self.root_dir = abspath(root_dir)
        self.train_file = join(self.root_dir, 'train.png')
        self.test_file = join(self.root_dir, 'test.png')
        self.data_file = self.train_file

    def __iter__(self):
        if self.mode == 'test':
            self.data_file = self.test_file
        # load images
        data_image = Image.open(self.data_file)
        
        # data augmentation
        resized_size = 128
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomRotation((-90, 90)),
            transforms.RandomResizedCrop(resized_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        input_image = torch.zeros(self.batch_size, 4, resized_size, resized_size)
        gt_label = torch.zeros(self.batch_size, 3, resized_size, resized_size)

        for i in range(self.batch_size):
            print("in loader", i)
            trsf_image = transform(data_image)
            mask = self.__generateMask(trsf_image.shape)
            input_image[i] = torch.cat((trsf_image, mask), 0)
            gt_label[i] = trsf_image
            yield (input_image, gt_label)
        
    def setMode(self, mode):
        self.mode = mode

    def __generateMask(self, shape, n_holes = 5):
        mask = torch.zeros(1,shape[1],shape[2])
        _, mask_h, mask_w = mask.shape
        masks = []
        for i in range(n_holes):
            p = random.uniform(0, 1)
            if(p>0.5):
                hole_w, hole_h = 64, 8
            else:
                hole_w, hole_h = 8, 64
            # random generate hole position
            # offset_x = random.randint(0, mask_w - hole_w)
            # offset_y = random.randint(0, mask_h - hole_h)
            offset_x = random.randint(1, mask_w - hole_w - 1) # rectangle not blurry on the boundary
            offset_y = random.randint(1, mask_h - hole_h - 1)
            mask[:, offset_y : offset_y + hole_h, offset_x : offset_x + hole_w] = 1.0
        return mask

# Test dataloader
loader = DataLoader()
for i, (img, label) in enumerate(loader):
    print("in test", i)
    figs, axes = plt.subplots(1, 2)
    axes[0].imshow(label[5].permute(1,2,0).numpy())
    axes[1].imshow(label[10].permute(1,2,0).numpy())
    plt.show()
    if(i==5): 
        break

