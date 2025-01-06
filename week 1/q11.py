import torch

# set the random seed for reproducibility
torch.manual_seed(7)

# create a tensor of shape (1, 1, 1, 10) with random values from a normal distribution
tensor = torch.randn(1, 1, 1, 10)
print(tensor)
print(tensor.shape)
print()

# remove dimensions of size 1, resulting in a tensor of shape (10)
squeeze_tensor = tensor.squeeze()
print(squeeze_tensor)
print(squeeze_tensor.shape)
