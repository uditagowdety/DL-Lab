# Apply torch.nn.Conv2d to the input image of Qn 1 with out-channel=3 and observe the
# output. Implement the equivalent of torch.nn.Conv2d using the torch.nn.functional.conv2D
# to get the same output. You may ignore bias.

import torch
import torch.nn.functional as f

image=torch.rand(6,6)
print("image =",image)

image=image.unsqueeze(dim=0)
print("image.shape =",image.shape)

image=image.unsqueeze(dim=0)
print("image.shape =",image.shape)

print("image =",image)

kernel=torch.rand(3,1,3,3)
# kernel=kernel.unsqueeze(dim=0)
# kernel=kernel.repeat(3,1,1,1)
print("kernel = ",kernel)

outImage=f.conv2d(image,kernel,stride=1,padding=0)

print("outImage =",outImage)
print("outImage.shape = ",outImage.shape)