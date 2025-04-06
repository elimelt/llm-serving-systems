

import torch

import triton
import triton.language as tl


@triton.jit
def multi_step_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    pid = tl.program_id(axis=0) 
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * 2 + x * y + y * y * y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


def multi_step(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    multi_step_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output



torch.manual_seed(0)
size = 1000000000
x = torch.rand(size, device="cuda")
y = torch.rand(size, device="cuda")
output_torch =  x * 2 + x * y + y * y
output_triton = multi_step(x, y)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100):
    output_triton = multi_step(x, y)
end.record()
torch.cuda.synchronize()
print(f'Triton kernel time: {start.elapsed_time(end) / 100 / 1000} s')
print(f"Bandwidth: {3 * size * x.element_size() / (start.elapsed_time(end) / 1000 / 100) / 1e9} GB/s")

print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
