import torch.nn as nn
import torch

def create_tensor(rows, cols):
    x = torch.zeros((rows, cols), dtype=torch.float32)
    for i in range(rows):
        for j in range(cols):
            x[i][j] = i * cols + j + 1
    return x
x = create_tensor(1, 10)
print(x)
# rms norm of x
rms_norm = nn.RMSNorm(normalized_shape=x.shape[1], eps=1e-6)
print(rms_norm(x))