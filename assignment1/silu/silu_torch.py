import torch

def silu_torch(x):
    return x * (1 / (1 + torch.exp(-x)))
  
  
if __name__ == "__main__":
    x = torch.tensor([1.0, 2.0, 3.0])
    output = silu_torch(x)
    print(output)