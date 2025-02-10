import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Custom Dataset
class RegressionDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X.view(-1, 1)
        self.Y = Y.view(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Define the Regression Model
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

# Given dataset
X = torch.tensor([12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4,
                  19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2])
Y = torch.tensor([11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6,
                  16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6])
dataset = RegressionDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Hyperparameters
learning_rate = 0.001
epochs = 100

# Initialize model
model = RegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
loss_values = []
for epoch in range(epochs):
    for batch_X, batch_Y in dataloader:
        optimizer.zero_grad()  # Zero gradients
        yp = model(batch_X)  # Forward pass
        loss = criterion(yp, batch_Y)  # Compute MSE loss
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
