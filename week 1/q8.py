import torch

tensor1=torch.rand(2,3)
tensor2=torch.rand(2,3)

tensor2_reshaped=tensor2.transpose(0,1)

result=torch.matmul(tensor1,tensor2_reshaped)
print(result.shape)