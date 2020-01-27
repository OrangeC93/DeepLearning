import torch

a = torch.randn(3,5)
b = torch.randn(3,5)

print(a)
print(b)

c = torch.cat((a,b),dim=1)
d = torch.cat((a,b),dim=0)

print(c)