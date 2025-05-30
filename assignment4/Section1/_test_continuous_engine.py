import torch
import time
import random
from typing import List
from continous_engine import Engine, Request

def generate_random_requests(engine: Engine, num_requests: int, max_input_len: int, max_output_len: int) -> List[Request]:
    """Generate random requests with uniform distribution of input and output lengths."""
    requests = []
    for i in range(num_requests):
        input_len = random.randint(1, max_input_len)
        output_len = random.randint(1, max_output_len)
        # Generate random token IDs for input
        input_ids = torch.randint(0, engine.tokenizer.vocab_size, (input_len,))
        requests.append(Request(i, input_ids, output_len))
    return requests

def naive_scheduling(engine: Engine, requests: List[Request], batch_size: int) -> float:
    """Process requests using naive scheduling (one batch at a time)."""
    start_time = time.time()
    
    # Process requests in batches
    for i in range(0, len(requests), batch_size):
        batch = requests[i:i + batch_size]
        # Process each request in the batch sequentially
        for req in batch:
            # Prefill phase
            engine.run([req], num_decode_req=0)
            # Decode phase
            for _ in range(req.output_length):
                engine.run([req], num_decode_req=1)
    
    end_time = time.time()
    return end_time - start_time

def continuous_batching(engine: Engine, requests: List[Request], batch_size: int) -> float:
    """Process requests using continuous batching."""
    start_time = time.time()
    
    # Initialize request queues
    prefill_queue = requests.copy()
    decode_queue = []
    completed_requests = 0
    
    while completed_requests < len(requests):
        # Fill decode queue up to batch size
        while len(decode_queue) < batch_size and prefill_queue:
            req = prefill_queue.pop(0)
            # Prefill phase
            engine.run([req], num_decode_req=0)
            decode_queue.append(req)
        
        if decode_queue:
            # Process decode batch
            batch = decode_queue[:batch_size]
            engine.run(batch, num_decode_req=len(batch))
            
            # Update decode queue
            new_decode_queue = []
            for req in decode_queue:
                if req.current_length < req.prompt_length + req.output_length:
                    new_decode_queue.append(req)
                else:
                    completed_requests += 1
            decode_queue = new_decode_queue
    
    end_time = time.time()
    return end_time - start_time

def main():
    # Test parameters
    num_requests = 1
    batch_size = 1
    max_input_len = 1024
    max_output_len = 1024
    
    # Initialize engine
    engine = Engine()
    
    # Generate random requests
    requests = generate_random_requests(engine, num_requests, max_input_len, max_output_len)
    
    # Run naive scheduling
    naive_time = naive_scheduling(engine, requests, batch_size)
    print(f"Naive scheduling time: {naive_time:.2f} seconds")
    
    # Reset engine for continuous batching
    engine = Engine()
    
    # Run continuous batching
    continuous_time = continuous_batching(engine, requests, batch_size)
    print(f"Continuous batching time: {continuous_time:.2f} seconds")
    
    # Calculate speedup
    speedup = naive_time / continuous_time
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main() 