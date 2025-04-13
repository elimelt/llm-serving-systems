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
    
    # Add the tensors
    for i in range(10):
        result = add_tensors(tensor1, tensor2)

    result = result.cpu()
    # Print the result
    print("Result of addition:", result)