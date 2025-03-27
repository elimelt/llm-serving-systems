import torch

num_rows = 8192 
num_cols = 8192

# Create a random tensor
x = torch.randn(num_rows, num_cols, dtype=torch.float32, device='cuda')
# Transpose the tensor
y = x.t()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
# Record the start time
start.record()

for i in range(1000):
    # Transpose the tensor
    y = x.t().contiguous()
# Record the end time
end.record()
# Wait for the events to be recorded
torch.cuda.synchronize()
# Calculate the elapsed time
elapsed_time = start.elapsed_time(end)
per_iteration_time = elapsed_time / 1000 / 1000
print(f"Elapsed time for each kernel: {per_iteration_time * 1000:.3f} ms")
print(f"Bandwidth {2 * x.element_size() * x.numel()\
    / per_iteration_time / 1e9:.3f} GB/s")
