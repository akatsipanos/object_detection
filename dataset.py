#%%
from PIL import Image
import numpy as np
import os
import glob
import torch
import torch.utils.data
import xmltodict
# from torchvision import datasets 

class FruitDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = glob.glob(os.path.join(self.root, '*.jpg'))
        self.info = glob.glob(os.path.join(self.root, '*.xml'))
    
    # def generate_data(self):
    #     data_path = self.info
    #     data = []
    #     target = {}
    #     boxes = []
    #     img_id = []
    #     labels = []

    #     for i in data_path:
    #         with open(i,'r') as f:
    #             a = f.read()
    #         b = xmltodict.parse(a)['annotation']
    #         for i in b['object']:
    #             boxes.append([int(b['object']['bndbox']['xmin']),
    #                         int(b['object']['bndbox']['ymin']),
    #                         int(b['object']['bndbox']['xmax']), 
    #                         int(b['object']['bndbox']['ymax'])])
    #     target['boxes'] = boxes
    #     return target

    def __getitem__(self,idx):
        # img_path = os.path.join(self.root, self.imgs[idx])
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')

        data_path = self.info
        data = []
        target = {}
        boxes = []
        img_id = []
        labels = []
        label_dict = {'apple':1, 'banana': 2, 'orange':3}

        with open(data_path[idx],'r') as f:
            a = f.read()

        b = xmltodict.parse(a)['annotation']

        if type(b['object']) is list:
            for i in b['object']:
                boxes.append([int(i['bndbox']['xmin']),
                              int(i['bndbox']['ymin']),
                              int(i['bndbox']['xmax']), 
                              int(i['bndbox']['ymax'])])
                labels.append(label_dict[i['name']])
                
        else:
            boxes.append([int(b['object']['bndbox']['xmin']),
                          int(b['object']['bndbox']['ymin']),
                          int(b['object']['bndbox']['xmax']), 
                          int(b['object']['bndbox']['ymax'])])
            labels.append(label_dict[b['object']['name']])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        target['boxes'] = boxes
        target['labels'] = torch.as_tensor(labels, dtype=torch.int16)
        target['image_id'] = torch.as_tensor(idx, dtype=torch.int16)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target['area'] = area

        return img, target
    
    def __len__(self):
        return len(self.imgs)
    
#%%
dataset = FruitDataset('data/train')
dataset[10]