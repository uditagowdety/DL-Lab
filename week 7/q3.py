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
    def __init__(self,with_dropout=True):
        super(CNN,self).__init__()
        if with_dropout:
            self.net=nn.Sequential(
                nn.Conv2d(3,16,3,1),
                nn.Conv2d(16,32,3,1),
                nn.Flatten(),
                nn.Linear(124*124*32,512),
                nn.Dropout(0.5),
                nn.Linear(512,2)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(3, 16, 3, 1),
                nn.Conv2d(16, 32, 3, 1),
                nn.Flatten(),
                nn.Linear(124 * 124 * 32, 512),
                nn.Linear(512, 2)
            )

    def forward(self,x):
        return self.net(x)

def epoch_train(model, train_loader, criterion, optimizer, epochs, loss_list, lambda_l1=0.01):
    for epoch in range(epochs):
        model.train()
        epoch_loss=0.0

        total_correct=0
        total_samples=0

        for images, labels in train_loader:
            optimizer.zero_grad()

            outputs=model(images)
            loss=criterion(outputs, labels)

            l1_penalty=0
            for param in model.parameters():
                l1_penalty+=torch.sum(torch.abs(param))

            loss.backward()
            optimizer.step()

            epoch_loss+=loss.item()

            _,predicted=torch.max(outputs,1)
            total_samples += labels.size(0)
            total_correct+=(predicted==labels).sum().item()

        accuracy=100*total_correct/total_samples
        loss_list.append(epoch_loss / len(train_loader))
        print(f"epoch {epoch + 1}/{epochs}, loss = {loss_list[-1]:.4f}, accuracy = {accuracy:.2f}%")

model_dropout=CNN(with_dropout=True)
model_no_dropout=CNN(with_dropout=False)
criterion=nn.CrossEntropyLoss()

optimizer=optim.SGD(model_dropout.parameters(),lr=0.01)

epochs=2
loss_manual=[]

print("training model with dropout regularization...")
epoch_train(model_dropout,train_loader,criterion,optimizer,epochs,loss_manual)

print("training model without dropout regularization...")
epoch_train(model_no_dropout,train_loader,criterion,optimizer,epochs,loss_manual)