import torch
from torch.tensor import Tensor

x = torch.ones(2, 2, requires_grad=True)
y = torch.add(x, Tensor([2]))
z = y * y * 3
out = z.mean()

print(z)
print(out)
out.backward()
x.grad