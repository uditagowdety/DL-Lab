import torch

tensor=torch.arange(9)
print(tensor)
print()

#reshaping a tensor:
reshaped_tensor=torch.reshape(tensor,(3,3))
print(reshaped_tensor)
print()

#viewing a tensor:
view_tensor=tensor.view(3,3)
print(view_tensor)
print()

#stacking 2 tensors:
tensor_1=torch.tensor([1,2])
tensor_2=torch.tensor([3,4])
stacked_tensor=torch.stack([tensor_1,tensor_2],dim=0)
print(stacked_tensor)
print()

#squeezing a tensor:
new_tensor=torch.zeros(1,2,3)
print(new_tensor)
print()
squeeze_tensor=torch.squeeze(new_tensor,dim=0)
print(squeeze_tensor)
print()

#unsqueezing a tensor
unsqueeze_tensor=torch.unsqueeze(tensor_1,dim=0)
print(unsqueeze_tensor)