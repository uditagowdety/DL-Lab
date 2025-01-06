import torch

# create a 1D tensor with values from 0 to 8
tensor = torch.arange(9)
print(tensor)
print()

# reshaping a tensor to a 3x3 matrix
reshaped_tensor = torch.reshape(tensor, (3, 3))
print(reshaped_tensor)
print()

# viewing a tensor (creating a new view of the same data without copying it)
view_tensor = tensor.view(3, 3)
print(view_tensor)
print()

# stacking 2 tensors along a new dimension (dim=0, creating a 2x2 tensor)
tensor_1 = torch.tensor([1, 2])
tensor_2 = torch.tensor([3, 4])
stacked_tensor = torch.stack([tensor_1, tensor_2], dim=0)
print(stacked_tensor)
print()

# squeezing a tensor (removes dimensions with size 1)
new_tensor = torch.zeros(1, 2, 3)  # tensor with shape (1, 2, 3)
print(new_tensor)
print()
squeeze_tensor = torch.squeeze(new_tensor, dim=0)  # removes the dimension with size 1 at index 0
print(squeeze_tensor)
print()

# unsqueezing a tensor (adds a new dimension with size 1 at the specified index)
unsqueeze_tensor = torch.unsqueeze(tensor_1, dim=0)  # adds a new dimension at index 0
print(unsqueeze_tensor)
