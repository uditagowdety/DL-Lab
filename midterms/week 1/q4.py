import torch

tensor=torch.randn(4,4)
print(tensor)

tensor_to_numpy=tensor.numpy()
print(tensor_to_numpy)

numpy_to_tensor=torch.from_numpy(tensor_to_numpy)
print(numpy_to_tensor)