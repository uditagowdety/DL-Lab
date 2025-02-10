#alexnet cats and dogs

import PIL.Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms
import glob
from torchvision.models import AlexNet_Weights

model=torch.hub.load("pytorch/vision:v0.10.0",model="alexnet",weights=AlexNet_Weights.DEFAULT)
batch_size=4

criterion=nn.CrossEntropyLoss
optimizer=torch.optim.SGD(model.parameters(),lr=0.001)

