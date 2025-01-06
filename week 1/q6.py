import torch

tensor1=torch.randn(1,3,5)
tensor2=torch.randn(1,7)

tensor2_reshaped=tensor2.view(1,7,1)

result=torch.matmul(tensor1,tensor2_reshaped)
print(result.shape)

#i dont really understand what this question means so ill deal with it later