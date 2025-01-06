import torch

# create a 2x3 tensor with random values
tensor = torch.rand(2, 3)
print(tensor)

# print the index of the maximum value in the tensor
print(torch.argmax(tensor))

# print the index of the minimum value in the tensor
print(torch.argmin(tensor))
