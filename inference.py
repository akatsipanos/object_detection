#%%
from dataset import FruitDataset 
from PIL import ImageDraw
import torchvision
import torch
from train import get_instance_segmentation_modified, get_transform

def show_prediction(idx, data_path):
    if 'train' in data_path:
        dataset = FruitDataset(data_path, get_transform(train=True))
    else:
        dataset = FruitDataset(data_path, get_transform(train=False))
    img_tensor,_ = dataset[idx]
    model = get_instance_segmentation_modified(num_classes=3)
    model.load_state_dict(torch.load('first_model.pt'))
    model.eval()
    device = torch.device('cpu')
    with torch.no_grad():
        pred = model([img_tensor.to(device)])

    transform = torchvision.transforms.ToPILImage()
    img = transform(img_tensor)
    img1 = ImageDraw.Draw(img)
    boxes = pred[0]['boxes']
    for i in range(len(boxes)):
        x0 = int(boxes[i][0])
        y0 = int(boxes[i][1] )
        x1 = int(boxes[i][2])
        y1 = int(boxes[i][3] )
        img1.rectangle((x0,y0,x1,y1), outline='red')

    img.show()
    return pred

if __name__ == '__main__':
    pred = show_prediction(idx=80, data_path='data/train')