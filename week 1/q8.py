import torch

# create two 2x3 tensors with random values
tensor1 = torch.rand(2, 3)
tensor2 = torch.rand(2, 3)

# transpose tensor2 from (2, 3) to (3, 2)
tensor2_reshaped = tensor2.transpose(0, 1)

# perform matrix multiplication between tensor1 (2x3) and tensor2_reshaped (3x2)
result = torch.matmul(tensor1, tensor2_reshaped)

# print the result shape
print(result.shape)
