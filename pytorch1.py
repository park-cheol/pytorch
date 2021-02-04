import torch
import numpy as np

x = torch.empty(5, 4)
print(x)

y = torch.ones(3, 2)
print(y)

y = y.view(6)
print(y)

x = torch.ones(1)
print(x.item())

x = torch.ones(2, 2, requires_grad=True)

backward() => chein rule
x.grad # 미분 후 보기