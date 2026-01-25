import torch

x = torch.tensor([0.1, 0.2, 0.3, 0.5])

y = torch.multinomial(x,1)

print(y)