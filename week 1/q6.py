import torch

# create a 7x7 tensor with random integer values between 0 and 9
tensor1 = torch.randint(0, 10, (7, 7))

# create a 1x7 tensor with random integer values between 0 and 9
tensor2 = torch.randint(0, 10, (1, 7))

# transpose tensor2, changing its shape from (1, 7) to (7, 1)
# this makes it suitable for matrix multiplication with tensor1
tensor2_reshaped = tensor2.transpose(0, 1)

# perform matrix multiplication between tensor1 (7x7) and tensor2_reshaped (7x1)
# the result will be a tensor of shape (7, 1) because the inner dimensions match (7)
result = torch.matmul(tensor1, tensor2_reshaped)

# print the shape of the result tensor
print(result.shape)
