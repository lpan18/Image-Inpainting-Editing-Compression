import os
from os.path import isdir, exists, abspath, join
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class DataLoader(Dataset):
    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
    
    def __len__(self):
        return len(self.dataset_list)    
    
    def __getitem__(self, idx):
        # load images
        img_path = self.dataset_list[idx]
        data_image = Image.open(img_path)

        # data augmentation
        resized_size = 128
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomRotation(5),
            transforms.RandomResizedCrop(resized_size,scale=(0.14, 0.17)),
            transforms.ToTensor()   # transforms the image to a tensor with range [0,1].
        ])
        gt_label = transform(data_image).float()
        mask = self.__generateMask(gt_label.shape)
        image_mask = (gt_label - gt_label * (1 - mask))
        input_image = torch.cat((image_mask, mask), 0).float()
        return (input_image, gt_label)
        
    def __generateMask(self, shape, n_holes = 5):
        mask = torch.ones(1,shape[1],shape[2])
        _, mask_h, mask_w = mask.shape
        masks = []
        for i in range(n_holes):
            p = random.uniform(0, 1)
            if(p>0.5):
                hole_w, hole_h = 64, 8
            else:
                hole_w, hole_h = 8, 64
            # random generate hole position
            offset_x = random.randint(1, mask_w - hole_w - 1) # rectangle not blurry on the boundary
            offset_y = random.randint(1, mask_h - hole_h - 1)
            mask[:, offset_y : offset_y + hole_h, offset_x : offset_x + hole_w] = 0
        return mask

if __name__ == '__main__':
    data_dir = 'data/train.png'
    dataset_list = []
    for i in range(16):
        dataset_list.append(data_dir)
    train_dataset = DataLoader(dataset_list)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=0)
    for i, (img, label) in enumerate(train_loader):
        figs, axes = plt.subplots(1, 3)
        img_mask = img[:,0:3,:,:]
        mask = img[:,3:4,:,:]
        axes[0].imshow(img_mask[0].permute(1,2,0).numpy())   # show first 3 channel
        axes[1].imshow(mask[0].squeeze(0).numpy(), cmap=cm.gray) # show mask channel
        # axes[0].imshow(img[0].squeeze(0).numpy())
        axes[2].imshow(label[0].permute(1,2,0).numpy()) # show label
        plt.show()
        break
