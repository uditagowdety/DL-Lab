import torch

# create a 1D tensor with random values between 0 and 1, of size 9
tensor = torch.rand(9)
print(tensor)

# accessing the element at index 4 (5th element) of the tensor
print(tensor[4])

# accessing the last element of the tensor using negative indexing
print(tensor[-1])

# accessing multiple elements at indices 1, 3, and 4 using a list of indices
print(tensor[[1, 3, 4]])
