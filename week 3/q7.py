import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Custom Dataset for Logistic Regression
class LogisticDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X.view(-1, 1)
        self.Y = Y.view(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Define the Logistic Regression Model
class LogisticModel(nn.Module):
    def __init__(self):
        super(LogisticModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input feature
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Given dataset
X = torch.tensor([1, 5, 10, 10, 25, 50, 70, 75, 100], dtype=torch.float32)
Y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32)
dataset = LogisticDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Hyperparameters
learning_rate = 0.001
epochs = 100

# Initialize model
model = LogisticModel()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
loss_values = []
for epoch in range(epochs):
    for batch_X, batch_Y in dataloader:
        optimizer.zero_grad()  # Zero gradients
        yp = model(batch_X)  # Forward pass
        loss = criterion(yp, batch_Y)  # Compute Binary Cross Entropy loss
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters
    
    loss_values.append(loss.item())
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")

# Plot epoch vs loss
plt.plot(range(epochs), loss_values, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.legend()
plt.show()

# Verify prediction for X=30
test_point = torch.tensor([[30]], dtype=torch.float32)
prediction = model(test_point).item()
print(f"Prediction for X=30: {prediction:.4f}")
