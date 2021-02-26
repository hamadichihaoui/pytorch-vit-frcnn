# --------------------------------------------------------
# Written by Hamadi Chihaoui at 1:37 PM 2/26/2021 
# --------------------------------------------------------
import torchvision.datasets as dset
import torch
from config import *
from training.engine import train_one_epoch, evaluate
from model.vit_frcnn import ViTRFCNN
import torchvision.transforms as transforms
import numpy as np
import torchvision.transforms.functional as F


class SquarePad:
       def __call__(self, image):
            w, h = image.size
            max_wh = np.max([w, h])
            hp = int((max_wh - w) / 2)
            vp = int((max_wh - h) / 2)
            padding = (hp, vp, hp, vp)
            return F.pad(image, padding, 0, 'constant')


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_transform = transforms.Compose([
    SquarePad(),
    transforms.Resize((800, 800)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    SquarePad(),
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
coco_train = dset.CocoDetection(root=path2traindata, annFile=path2trainjson, transforms=train_transform)
coco_val = dset.CocoDetection(root=path2valdata, annFile=path2valjson, transforms= val_transform)


def collate_fn(batch):
    return tuple(zip(*batch))


train_data_loader = torch.utils.data.DataLoader(coco_train, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_data_loader = torch.utils.data.DataLoader(coco_val, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)

# For Training

model = ViTRFCNN(num_classes=92).to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 2
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_data_loader, device, epoch, 10)
    coco_evaluator = evaluate(model, val_data_loader, device)


