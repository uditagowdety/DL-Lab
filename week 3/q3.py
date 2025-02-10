import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the Regression Model
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.w = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        return self.w * x + self.b
    
    def update(self, w_grad, b_grad, learning_rate):
        self.w.data -= learning_rate * w_grad
        self.b.data -= learning_rate * b_grad
    
    def reset_grad(self):
        if self.w.grad is not None:
            self.w.grad.zero_()
        if self.b.grad is not None:
            self.b.grad.zero_()

# Given dataset
X = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0]).view(-1, 1)
Y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0]).view(-1, 1)

# Hyperparameters
learning_rate = torch.tensor(0.001)
epochs = 100

# Initialize model
model = RegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate.item())

# Training loop
loss_values = []
for epoch in range(epochs):
    optimizer.zero_grad()  # Zero gradients
    yp = model(X)  # Forward pass
    loss = criterion(yp, Y)  # Compute MSE loss
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
