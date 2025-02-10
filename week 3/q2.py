import torch
import torch.nn as nn
import torch.optim as optim

# Given dataset
X = torch.tensor([2.0, 4.0]).view(-1, 1)  # Reshaped for PyTorch
Y = torch.tensor([20.0, 40.0]).view(-1, 1)

# Model parameters
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

learning_rate = 0.001
optimizer = optim.SGD([w, b], lr=learning_rate)

# Perform two epochs
for epoch in range(2):
    optimizer.zero_grad()  # Zero gradients
    yp = w * X + b  # Predicted values
    loss = ((yp - Y) ** 2).mean()  # Mean Squared Error (MSE)
    loss.backward()  # Compute gradients
    optimizer.step()  # Update parameters
    
    # Print results
    print(f"Epoch {epoch+1}:")
    print(f"w_grad: {w.grad.item():.6f}, b_grad: {b.grad.item():.6f}")
    print(f"Updated w: {w.item():.6f}, Updated b: {b.item():.6f}")
    print(f"Error (yp - y): {(yp - Y).detach().numpy()}")
    print("-" * 40)