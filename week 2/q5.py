import torch

x=torch.arange(5,dtype=torch.float32,requires_grad=True)
print(x)

def f(x):
    return 8*x**4+3*x**3+7*x**2+6*x+3

y=f(x)

y.backward(torch.ones_like(x))

print("Gradients of y with respect to x:", x.grad)


def der(x):
    return 32*x**3+9*x**2+14*x+6

derivs=[der(i) for i in x]
print(derivs)