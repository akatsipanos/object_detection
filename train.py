import numpy as np
#%%
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from dataset import FruitDataset
from engine import train_one_epoch, evaluate
import utils
import transforms as T

def get_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

dataset_train = FruitDataset('data/train', get_transform(train=True)) # add transform part
dataset_test = FruitDataset('data/test', get_transform(train=False)) # add transform part

#%%

torch.manual_seed(1)
data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                                batch_size = 8,
                                                shuffle = True,
                                                collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                batch_size = 8,
                                                shuffle = True,
                                                collate_fn=utils.collate_fn)

device = torch.device('cpu')
num_classes = 3
model = get_instance_segmentation(num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 10
for epoch in range(num_epochs):
    train_one_epoch(model,optimizer, data_loader_train, device, epoch, print_freq=10)
    lr_scheduler.step()
    evaluate(model, data_loader_test, device=device)
    print(f'Finished training epoch {epoch}')

#%%