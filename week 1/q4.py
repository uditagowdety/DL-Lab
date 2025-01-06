import torch
import numpy as np

np_array=np.array([1,2,3,4,5])
np_to_tensor=torch.from_numpy(np_array)
print(np_array)
print(np_to_tensor)

tensor=torch.arange(5)
tensor_to_np=tensor.numpy()
print(tensor)
print(tensor_to_np)