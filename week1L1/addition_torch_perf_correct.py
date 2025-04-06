import time
import torch

def add_tensors(a, b):
    return a + b

if __name__ == "__main__":
    
    num_elements = 10**9
    
    # Create two tensors with 1e9 elements each
    tensor1 = torch.rand(num_elements, device='cpu')
    tensor2 = torch.rand(num_elements, device='cpu')
    
    tensor1 = tensor1.to('cuda')
    tensor2 = tensor2.to('cuda')
    
    add_tensors(tensor1, tensor2)
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    # Add the tensors
    for i in range(100):
        result = add_tensors(tensor1, tensor2)
    end.record()
    torch.cuda.synchronize()

    each_iter_time = start.elapsed_time(end) / 1000 / 100  # Convert milliseconds to seconds
    
    print("Time taken for addition:", each_iter_time, "seconds")
    print("Bandwidth: ", 3 * tensor1.element_size() * tensor1.numel() / each_iter_time / 1e9, "GB/s")

    result = result.cpu()
    # Print the result
    print("Result of addition:", result)