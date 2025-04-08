import torch
import triton
import triton.language as tl


@triton.jit
def silu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) 
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * (1 / (1 + tl.exp(-x)))
    tl.store(output_ptr + offsets, output, mask=mask)
    

def silu_triton(x):
    
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    silu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output