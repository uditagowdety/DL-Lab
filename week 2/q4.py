import torch

x=torch.arange(9,dtype=torch.float32,requires_grad=True)
print(x)

def f(x):
    return torch.exp(-x**2-2*x-torch.sin(x))

y=f(x)

y.backward(torch.ones_like(x))

print("Gradients of y with respect to x:", x.grad)


def der(x):
    return (-2*x-2-torch.cos(x))*torch.exp(-x**2-2*x-torch.sin(x))

derivs=[der(i) for i in x]
print(derivs)