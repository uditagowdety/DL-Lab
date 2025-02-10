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

# Prepare Data:
#     Load cats & dogs dataset (train/val).
#     Preprocess: resize images to 224x224, normalize, and apply data augmentation (flip, crop).
#
# Load Pre-trained AlexNet:
#     Use AlexNet pretrained on ImageNet.
#     Modify final layer for binary classification (2 classes: cat, dog).
#
# Setup Training:
#     Move model to device (GPU if available).
#     Choose loss function (Cross-Entropy) and optimizer (Adam).
#
# Train Model:
#     Loop through epochs:
#         Train on batches (compute loss, backpropagate).
#         Validate after each epoch (track accuracy, loss).
#
# Save Best Model:
#     Track validation accuracy; save model if it improves.
#
# Test Model:
#     Test with new images or saved model on the validation set.