import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root="./cats_and_dogs_filtered/train", transform=transform)
test_dataset = datasets.ImageFolder(root="./cats_and_dogs_filtered/validation", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class CustomDropout(nn.Module):
    def __init__(self,rate):
        super(CustomDropout,self).__init__()
        self.rate=rate

    def forward(self,x):
        if self.training:
            mask=(torch.rand_like(x)<(1-self.rate)).float()
            x=x*mask/(1-self.rate)
        return x

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN,self).__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,16,3,1),
            CustomDropout(0.5),
            nn.Conv2d(16,32,3,1),
            nn.Flatten(),
            nn.Linear(124*124*32,512),
            nn.Linear(512,2)
        )

    def forward(self,x):
        return self.net(x)

class BuiltInCNN(nn.Module):
    def __init__(self):
        super(BuiltInCNN,self).__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,16,3,1),
            nn.Dropout(0.5),
            nn.Conv2d(16,32,3,1),
            nn.Flatten(),
            nn.Linear(124*124*32,512),
            nn.Linear(512,2)
        )

    def forward(self,x):
        return self.net(x)


def train(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

        accuracy = 100 * total_correct / total_samples
        print(f"epoch {epoch + 1}/{epochs}, loss: {epoch_loss / len(train_loader):.4f}, accuracy: {accuracy:.2f}%")


def test(model, test_loader):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    accuracy = 100 * total_correct / total_samples
    return accuracy

custom_model=CustomCNN()
builtin_model=BuiltInCNN()

criterion=nn.CrossEntropyLoss()
optimizer_custom = optim.SGD(custom_model.parameters(), lr=0.01)
optimizer_builtin = optim.SGD(builtin_model.parameters(), lr=0.01)

epochs = 5
print("training custom model with custom dropout...")
train(custom_model, train_loader, criterion, optimizer_custom, epochs)

print("training model with built-in dropout...")
train(builtin_model, train_loader, criterion, optimizer_builtin, epochs)

print("\ntesting custom model with custom dropout...")
custom_model_accuracy = test(custom_model, test_loader)
print(f"custom model accuracy: {custom_model_accuracy:.2f}%")

print("\ntesting model with built-in dropout...")
builtin_model_accuracy = test(builtin_model, test_loader)
print(f"built-in model accuracy: {builtin_model_accuracy:.2f}%")