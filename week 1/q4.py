import torch
import numpy as np

# create a numpy array with values from 1 to 5
np_array = np.array([1, 2, 3, 4, 5])

# convert the numpy array to a tensor using torch.from_numpy()
np_to_tensor = torch.from_numpy(np_array)
print("Numpy Array:")
print(np_array)

print("Converted Tensor:")
print(np_to_tensor)

# create a tensor with values from 0 to 4
tensor = torch.arange(5)

# convert the tensor back to a numpy array using tensor.numpy()
tensor_to_np = tensor.numpy()
print("Tensor:")
print(tensor)

print("Converted Numpy Array:")
print(tensor_to_np)
