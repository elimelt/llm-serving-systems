import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

SHAPE = (8192, 8192)
N_ITR = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def silu_torch(x):
  return x * (1 / (1 + torch.exp(-x)))

def torch_profiled():
  x = torch.randn(SHAPE)
  x = x.to(device)
  
  with profile(
      activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
      profile_memory=True,
      with_stack=True,
      record_shapes=True) as prof:
    with record_function("silu_torch"):
      for n_ier in range(N_ITR):
        silu_torch(x)
  
  prof.export_chrome_trace("torch_silu.json")
  print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
  
def unprofiled():
  x = torch.randn(SHAPE)
  x = x.to(device)
  
  for n_ier in range(N_ITR):
    silu_torch(x)
  
if __name__ == "__main__":
  # torch_profiled()
  unprofiled()
    