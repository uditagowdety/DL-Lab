import torch
from matplotlib import pyplot as plt

# Create the tensors x and y. They are the training examples in the dataset for linear regression
x = torch.tensor([12.4, 13.4, 14.5, 14.9, 16.5, 16.9, 15.5, 14.7, 17.9, 18.8, 20.3, 22.4, 19.4, 15.5, 16.3, 17.8, 19.2, 17.4, 19.5, 19.7, 21.2])
y = torch.tensor([11.2, 12.5, 13.2, 13.1, 14.1, 14.8, 14.3, 14.9, 15.6, 16.4, 17.7, 19.6, 19.4, 14.6, 15.1, 16.5, 16.1, 16.8, 15.2, 17.2, 18.6])

# The parameters to be learnt w, and b in the prediction y_p = wx + b
b = torch.rand(1, requires_grad=True)
w = torch.rand(1, requires_grad=True)

print("The parameters are {}, and {}".format(w, b))

# The Learning rate is set to alpha = 0.001
learning_rate = torch.tensor(0.001)

# The list of loss values for the plotting purpose
loss_list = []

# Run the training loop for N epochs
for epochs in range(100):
    loss = 0.0  # Compute the average loss for the training samples

    # Accumulate the loss for all the samples
    for i in range(len(x)):
        a = w * x[i] + b  # Compute prediction
        loss += (a - y[i]) ** 2  # Compute squared error

    # Find the average loss
    loss = loss / len(x)

    # Add the loss to a list for plotting
    loss_list.append(loss.item())

    # Compute the gradients using backward
    loss.backward()

    # Update the weight based on gradient descent
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Reset the gradients for the next epoch
    w.grad.zero_()
    b.grad.zero_()

    # Display the parameters and loss
    print("The parameters are w={},  b={}, and loss={}".format(w, b, loss.item()))

# Display the loss plot
plt.plot(loss_list)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Reduction over Training")
plt.show()
