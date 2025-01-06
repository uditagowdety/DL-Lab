import torch

# create two 2x3 tensors with random values between 0 and 1
tensor1 = torch.rand(2, 3)
tensor2 = torch.rand(2, 3)

# check if CUDA (GPU support) is available; if it is, set the device to 'cuda', else use 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"

# print the device (either 'cuda' or 'cpu')
print(device)

# move tensor1 to the GPU (if available), otherwise it stays on the CPU
tensor1_gpu = tensor1.to(device)

# print tensor1 after it has been moved to the selected device (CPU or GPU)
print(tensor1_gpu)

# move tensor2 to the GPU (if available), otherwise it stays on the CPU
tensor2_gpu = tensor2.to(device)

# print tensor2 after it has been moved to the selected device (CPU or GPU)
print(tensor2_gpu)
