import torch

torch.manual_seed(7)

tensor=torch.randn(1,1,1,10)
print(tensor)
print(tensor.shape)
print()

squeeze_tensor=tensor.squeeze()
print(squeeze_tensor)
print(squeeze_tensor.shape)