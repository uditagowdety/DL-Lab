import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

transform=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
])

train_dataset=datasets.ImageFolder(root="./cats_and_dogs_filtered/train", transform=transform)
test_dataset=datasets.ImageFolder(root="./cats_and_dogs_filtered/validation", transform=transform)

train_loader=data.DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=data.DataLoader(test_dataset,batch_size=32,shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,16,3,1),
            nn.Conv2d(16,32,3,1),
            nn.Flatten(),
            nn.Linear(124*124*32,512),
            nn.Linear(512,2)
        )

    def forward(self,x):
        return self.net(x)


def train_weight_decay(model,train_loader,criterion,optimizer,epochs,loss_list):
    for epoch in range(epochs):
        model.train()
        epoch_loss=0.0

        for images,labels in train_loader:
            optimizer.zero_grad()

            outputs=model(images)
            loss=criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss+=loss.item()

        loss_list.append(epoch_loss/len(train_loader))
        print(f"epoch {epoch+1}/{epochs}, loss = {loss_list[-1]:.4f}")

def train_manual(model, train_loader, criterion, optimizer, epochs, loss_list, lambda_l2=0.01):
    for epoch in range(epochs):
        model.train()
        epoch_loss=0.0

        for images, labels in train_loader:
            optimizer.zero_grad()

            outputs=model(images)
            loss=criterion(outputs, labels)

            l2_penalty=0
            for param in model.parameters():
                l2_penalty+=torch.sum(param**2)

            total_loss=loss+lambda_l2*l2_penalty
            total_loss.backward()
            optimizer.step()

            epoch_loss+=total_loss.item()

        loss_list.append(epoch_loss / len(train_loader))
        print(f"epoch {epoch + 1}/{epochs}, loss with l2 penalty = {loss_list[-1]:.4f}")

model=CNN()
criterion=nn.CrossEntropyLoss()

optimizer_weight_decay=optim.SGD(model.parameters(),lr=0.01,weight_decay=0.001)
optimizer_no_weight_decay=optim.SGD(model.parameters(),lr=0.01)
epochs=10
loss_wd=[]
loss_manual=[]

print("training model with optimizer weight decay...")
train_weight_decay(model,train_loader,criterion,optimizer_weight_decay,epochs,loss_wd)

print("training model with manual L2 regularization...")
train_manual(model,train_loader,criterion,optimizer_weight_decay,epochs,loss_manual)