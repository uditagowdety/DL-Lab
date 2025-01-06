import torch

# create a 2x3 tensor with random values
tensor = torch.rand(2, 3)
print(tensor)

# print the maximum value in the tensor
print(torch.max(tensor))

# print the minimum value in the tensor
print(torch.min(tensor))
