import torch

tensor=torch.randn(5,5)
print(tensor)
print("max: ",torch.max(tensor))
print("min: ",torch.min(tensor))
max_index=torch.argmax(tensor)
print("index of max: ",torch.unravel_index(max_index,tensor.shape))
min_index=torch.argmin(tensor)
print("index of max: ",torch.unravel_index(min_index,tensor.shape))
