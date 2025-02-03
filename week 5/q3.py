from pyexpat import features

import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.nn import Sequential
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.xpu import device

transform=transforms.Compose([transforms.ToTensor()])

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print("Type is = ", type(mnist_train))
print("Train Dataset size = ", len(mnist_train))

batch_size = 1

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size)

class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),  # Correct Conv2d layer
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2)
        )

        # The number of input features to the fully connected layer should depend on the output of the last Conv2D layer.
        # Assuming the input image is 28x28, after 3 MaxPool layers, the output size is 3x3.
        self.classification_head = nn.Sequential(
            nn.Linear(64 * 3 * 3, 20),  # 64 channels, 3x3 feature map (after 3 pooling layers)
            nn.ReLU(),
            nn.Linear(20, 10)  # 10 output classes (digits 0-9)
        )

    def forward(self,x):
        features=self.net(x)
        return self.classification_head(features.view(batch_size,-1))


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=CNNClassifier().to(device)
criterion=nn.CrossEntropyLoss()
optim=torch.optim.SGD(model.parameters(),lr=0.001)

def train_one_epoch(epoch_idx):
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs,labels=inputs.to(device),labels.to(device)
        optim.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = (total_correct / total_samples) * 100
    return total_loss / len(train_loader), accuracy


epochs = 15
loss_values = []
print("Initializing training on dev = ", device)
for epoch in range(epochs):
    avg_loss, accuracy = train_one_epoch(epoch)
    loss_values.append(avg_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

plt.plot(loss_values)
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()