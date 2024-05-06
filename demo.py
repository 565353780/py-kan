import torch

from py_kan.Model.KAN import KAN

model = KAN(width=[22, 11, 4], grid=3, k=3)

a = torch.rand(400, 22)

b = model(a)

print(b.shape)
