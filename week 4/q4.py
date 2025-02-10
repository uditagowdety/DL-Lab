import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the Feed Forward Neural Network
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

# Transform the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoader for training and testing
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model
model = FeedForwardNN()

# Set the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Zero the gradients before backward pass

        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss

        loss.backward()  # Backward pass (compute gradients)
        optimizer.step()  # Update the weights

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# Testing loop
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # No need to compute gradients during testing
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")


torch.save(model.state_dict(), "./model_state_dict.pth")
