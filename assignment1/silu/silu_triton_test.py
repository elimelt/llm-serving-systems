import torch
from silu_torch import silu_torch
from silu_triton_kernel import silu_triton

if __name__ == "__main__":
  torch.manual_seed(0)
  size = (8192, 8192)
  x = torch.rand(size, device="cuda")
  output_torch =  silu_torch(x)
  output_triton = silu_triton(x)

  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  start.record()
  for _ in range(100):
      output_triton = silu_triton(x)
  end.record()
  torch.cuda.synchronize()
  
  total_size = x.numel() * x.element_size()
  
  print(f'Triton kernel time: {start.elapsed_time(end) / 100 / 1000} s')
  print(f"Bandwidth: {2 * total_size / (start.elapsed_time(end) / 1000 / 100) / 1e9} GB/s")

  print(output_torch)
  print(output_triton)
  print(f'Passes? {torch.allclose(output_torch, output_triton)}')
