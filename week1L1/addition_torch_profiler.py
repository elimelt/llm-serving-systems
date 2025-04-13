import torch
from torch.profiler import profile, record_function, ProfilerActivity

def add_tensors(a, b):
    return a + b

if __name__ == "__main__":
    
    num_elements = 10**9
    
    # Create two tensors with 1e9 elements each
    tensor1 = torch.rand(num_elements, device='cpu')
    tensor2 = torch.rand(num_elements, device='cpu')
    
    tensor1 = tensor1.to('cuda')
    tensor2 = tensor2.to('cuda')
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=True,
        record_shapes=True,
    ) as prof:
        # Use record_function to mark the section of code to be profiled
        with record_function("add_tensors"):
            # Add the tensors
            for i in range(10):
                result = add_tensors(tensor1, tensor2)
    
    # export to chrome trace
    prof.export_chrome_trace("trace.json")

    result = result.cpu()
    # Print the result
    print("Result of addition:", result)