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
    
    t_start = time.time()
    # Add the tensors
    for i in range(100):
        result = add_tensors(tensor1, tensor2)
    t_end = time.time()
    # Print the time taken for the addition
    each_iter_time = (t_end - t_start) / 100
    print("Time taken for addition:", each_iter_time, "seconds")
    print("Bandwidth: ", tensor1.element_size() * tensor1.numel() / each_iter_time / 1e9, "GB/s")

    result = result.cpu()
    # Print the result
    print("Result of addition:", result)