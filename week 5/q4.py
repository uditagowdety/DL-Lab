import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Define the original CNN model
class CNN(nn.Module):
    def __init__(self, reduced=False):
        super(CNN, self).__init__()
        if not reduced:
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
        else:  # Reduced parameters version
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(32 * 7 * 7, 64)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc2 = nn.Linear(128 if not reduced else 64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to train the model
def train_model(model, trainloader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}')

# Function to evaluate the model
def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
    accuracy = 100 * correct / total
    return accuracy, confusion_matrix(all_labels, all_preds)

# Train and evaluate both models
original_model = CNN(reduced=False)
reduced_model = CNN(reduced=True)

print("Training Original Model")
train_model(original_model, trainloader)
original_accuracy, original_cm = evaluate_model(original_model, testloader)
original_params = sum(p.numel() for p in original_model.parameters())

print("Training Reduced Model")
train_model(reduced_model, trainloader)
reduced_accuracy, reduced_cm = evaluate_model(reduced_model, testloader)
reduced_params = sum(p.numel() for p in reduced_model.parameters())

# Calculate percentage drop in parameters
param_drop = (original_params - reduced_params) / original_params * 100
accuracy_drop = original_accuracy - reduced_accuracy

# Plot percentage drop in parameters vs accuracy
plt.figure(figsize=(6,4))
plt.plot(param_drop, accuracy_drop, 'ro', markersize=10)
plt.xlabel('Percentage Drop in Parameters')
plt.ylabel('Drop in Accuracy')
plt.title('Parameters vs Accuracy')
plt.grid()
plt.show()

# Print results
print(f'Original Model Parameters: {original_params}')
print(f'Reduced Model Parameters: {reduced_params}')
print(f'Parameter Reduction: {param_drop:.2f}%')
print(f'Original Accuracy: {original_accuracy:.2f}%')
print(f'Reduced Accuracy: {reduced_accuracy:.2f}%')

# Plot confusion matrix for reduced model
plt.figure(figsize=(6,6))
sns.heatmap(reduced_cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Reduced Model')
plt.show()
