import torch

original=torch.randn(1,3,5)
print(f"original tensor dimensions:  {original.shape}")

permuted=original.permute(1,0,2)
print(f"permuted tensor dimensions:  {permuted.shape}")
