import torch
import numpy as np


x, y, z = torch.zeros(3,3), torch.ones(3,3), torch.rand(3,3)
print(x,y,z)
print(x+y)
print(y @ z)
print(z.int())
print(z.numpy())

if torch.cuda.is_available():
    print("CUDA is available")
    y, z = y.cuda(), z.cuda()
    print(y @ z)

