import torch.nn as nn
import torch

x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32).T
print(x)
# rms norm of x
rms_norm = nn.RMSNorm(2)
print(rms_norm(x))