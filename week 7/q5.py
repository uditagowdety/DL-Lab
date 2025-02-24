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

def train_es(model,train_loader, val_loader,criterion,optimizer,epochs, patience):
    best_val_loss=float("inf")
    static_epochs=0

    for epoch in range(epochs):
        model.train()
        epoch_loss=0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            epoch_loss+=loss.item()

            model.eval()
            val_loss=0.0
            total_correct=0
            total_samples=0

            with torch.no_grad():
                for images, labels in val_loader:
                    outputs=model(images)
                    loss=criterion(outputs,labels)
                    val_loss+=loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total_samples += labels.size(0)
                    total_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        accuracy = 100 * total_correct / total_samples
        print(f"epoch {epoch + 1}/{epochs}, train loss: {epoch_loss / len(train_loader):.4f}, validation loss: {val_loss:.4f}, accuracy: {accuracy:.2f}%")

        if val_loss<best_val_loss:
            best_val_loss=val_loss
            static_epochs=0
        else:
            static_epochs+=1

        if static_epochs>=patience:
            print(f"early stopping at epoch {epoch+1}")
            break

def train_no_es(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        val_loss /= len(test_loader)
        accuracy = 100 * total_correct / total_samples
        print(f"epoch {epoch+1}/{epochs}, train loss: {epoch_loss/len(train_loader):.4f}, validation loss: {val_loss:.4f}, accuracy: {accuracy:.2f}%")


model_es=CNN()
model_no_es=CNN()

criterion = nn.CrossEntropyLoss()
optimizer_with_early_stopping = optim.SGD(model_es.parameters(), lr=0.01)
optimizer_without_early_stopping = optim.SGD(model_no_es.parameters(), lr=0.01)

epochs = 10
patience = 3  # Number of epochs to wait before stopping if no improvement in validation loss

print("training model with early stopping...")
train_es(model_es, train_loader, test_loader, criterion, optimizer_with_early_stopping, epochs, patience)

print("\ntraining model without early stopping...")
train_no_es(model_no_es, train_loader, criterion, optimizer_without_early_stopping, epochs)

