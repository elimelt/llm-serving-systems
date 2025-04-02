from re import S
from networkx import star_graph
import torch

num = 1024 * 1024 * 1024
# Create a tensor of size num of int32
a = torch.empty(num, dtype=torch.int32, device='cuda')
print(a)
b = torch.sum(a)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for i in range(100):
    b = torch.sum(a)
end.record()
torch.cuda.synchronize()
t_all = start.elapsed_time(end)
iteration_time = t_all / 100
print(f"Per iteration time: {iteration_time} ms")