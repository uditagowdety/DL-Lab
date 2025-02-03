# Implement convolution operation for a sample image of shape (H=6, W=6, C=1) with a
# random kernel of size (3,3) using torch.nn.functional.conv2d.

# What is the dimension of the output image? Apply, various values for parameter stride=1
# and note the change in the dimension of the output image. Arrive at an equation for the
# output image size with respect to the kernel size and stride and verify your answer with
# code. Now, repeat the exercise by changing padding parameter. Obtain a formula using
# kernel, stride, and padding to get the output image size. What is the total number of
# parameters in your network? Verify with code.

import torch
import torch.nn.functional as f

image=torch.rand(6,6)
print("image =",image)

image=image.unsqueeze(dim=0)
print("image.shape =",image.shape)

image=image.unsqueeze(dim=0)
print("image.shape =",image.shape)

print("image =",image)

kernel=torch.ones(3,3)
kernel=kernel.unsqueeze(dim=0)
kernel=kernel.unsqueeze(dim=0)

outImage=f.conv2d(image,kernel,stride=1,padding=0)

print("outImage =",outImage)
print("outImage.shape = ",outImage.shape)