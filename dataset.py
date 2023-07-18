#%%
from PIL import Image
import numpy as np
import os
import glob
import torch
import torch.utils.data
import xmltodict

class FruitDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = glob.glob(os.path.join(self.root, '*.jpg'))
        self.info = glob.glob(os.path.join(self.root, '*.xml'))
    
    def __getitem__(self,idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')

        data_path = self.info
        target = {}
        boxes = []
        labels = []
        label_dict = {'apple':0, 'banana': 1, 'orange':2}

        with open(data_path[idx],'r') as f:
            a = f.read()

        b = xmltodict.parse(a)['annotation']

        num_objs = 0
        if type(b['object']) is list:
            for i in b['object']:
                boxes.append([int(i['bndbox']['xmin']),
                              int(i['bndbox']['ymin']),
                              int(i['bndbox']['xmax']), 
                              int(i['bndbox']['ymax'])])
                labels.append(label_dict[i['name']])
                num_objs+=1

        else:
            boxes.append([int(b['object']['bndbox']['xmin']),
                          int(b['object']['bndbox']['ymin']),
                          int(b['object']['bndbox']['xmax']), 
                          int(b['object']['bndbox']['ymax'])])
            labels.append(label_dict[b['object']['name']])
            num_objs = 1

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        target['boxes'] = boxes
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        target['image_id'] = torch.as_tensor(idx, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target['area'] = area
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    def __len__(self):
        return len(self.imgs)
    
#%%
if __name__ == '__main__':
    dataset = FruitDataset('data/train')
    dataset[0]