import torch.nn as nn
import torch

def create_tensor(rows, cols):
    x = torch.zeros((rows, cols), dtype=torch.float32)
    for i in range(rows):
        for j in range(cols):
            x[i][j] = i * cols + j + 1
    return x
x = create_tensor(5, 5).T
print(x)
# rms norm of x
rms_norm = nn.RMSNorm(5)
print(rms_norm(x))