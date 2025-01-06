import torch

# create a tensor with random values of shape (1, 3, 5)
original = torch.randn(1, 3, 5)
print(f"original tensor dimensions:  {original.shape}")

# permuting the dimensions of the tensor. This swaps axes 0 and 1.
# the new shape becomes (3, 1, 5)
permuted = original.permute(1, 0, 2)
print(f"permuted tensor dimensions:  {permuted.shape}")
