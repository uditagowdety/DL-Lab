#fashion mnist

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

class FeedForwardNN(nn.Module):
    def __init__(self):
        super(FeedForwardNN, self).__init__()
        # Input layer: 784 features (28*28)
        # First hidden layer: 128 neurons
        self.fc1 = nn.Linear(784, 128)
        # Second hidden layer: 64 neurons
        self.fc2 = nn.Linear(128, 64)
        # Output layer: 10 classes (digits 0-9)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # Flatten the input image (28x28 -> 784)
        x = x.view(-1, 784)
        # First hidden layer with ReLU activation
        x = torch.relu(self.fc1(x))
        # Second hidden layer with ReLU activation
        x = torch.relu(self.fc2(x))
        # Output layer (raw logits)
        x = self.fc3(x)
        return x

transform=transforms.Compose([
    transforms.Resize((28,28)),
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])
])

train_dataset=datasets.FashionMNIST(root="./data",train=True, download=True,transform=transform)
test_dataset=datasets.FashionMNIST(root="./data",train=False, download=True,transform=transform)

train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False)

# model=models.resnet18(pretrained=True)
# model.conv1=nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
model = FeedForwardNN()  # Define the model class before loading
model.load_state_dict(torch.load("./model_state_dict.pth"))

# model.fc=nn.Linear(model.fc.in_features,10)

for param in model.parameters():
    param.requires_grad=False

for param in model.fc3.parameters():
    param.requires_grad=True

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.fc3.parameters(),lr=0.001)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=model.to(device)

epochs=10

for epoch in range(epochs):
    model.train()
    current_loss=0.0
    correct=0
    total=0

    for images, labels in train_loader:
        images, labels=images.to(device),labels.to(device)
        optimizer.zero_grad()

        outputs=model(images)
        loss=criterion(outputs,labels)

        loss.backward()
        optimizer.step()

        current_loss+=loss.item()

        x,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()

    epoch_loss=current_loss/len(train_loader)
    epoch_accuracy=correct*100/total
    print(f"epoch {epoch+1}/{epochs}, loss:{epoch_loss:.4f}, accuracy: {epoch_accuracy:.2f}%")

model.eval()
correct=0
total=0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels=images.to(device),labels.to(device)

        outputs=model(images)
        x,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%\n")

print("model's state dict:")
for param_tensor in model.state_dict().keys():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
print()