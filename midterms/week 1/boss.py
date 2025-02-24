import torch

tensor=torch.randn(5,5)

print("second row:",tensor[1])
print("third column: ", tensor[:,2])
print("tensor[4][2]: ", tensor[4][2])

tensor[4]=0

tensor[:,0]= 1
 
tNum=tensor.numpy()
nTen=torch.from_numpy(tNum)

mean=torch.mean(tensor)
print("mean: ",mean.item())
