import torch

tensor1=torch.rand(2,3)
tensor2=torch.rand(2,3)

device="cuda" if torch.cuda.is_available() else "cpu"
print(device)

tensor1_gpu=tensor1.to(device)
print(tensor1_gpu)

tensor2_gpu=tensor2.to(device)
print(tensor2_gpu)