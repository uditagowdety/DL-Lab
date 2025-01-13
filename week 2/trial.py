import torch

x=torch.tensor([2.,3.],requires_grad=True)
y=3*x**2

y.backward(torch.ones_like(x))

print(x.grad)