from tqdm import tqdm
from continous_engine import Engine
from continous_scheduler import Scheduler, InputRequest
import numpy as np
import torch
import gc

def example_run():
    engine = Engine()
    scheduler = Scheduler(engine, req_batch_size=128)

    sample_prompts = ["Today is a rainy day"] * 128 + ["UW is"] * 128

    # Enqueue and run
    for prompt in sample_prompts:
        scheduler.add_req(InputRequest(prompt, output_len=100))
        scheduler.run()

    # Drain remaining requests
    while not scheduler.finished():
        scheduler.run()

    scheduler.print_completed()

def sample_uniform_ints(l, min=1, max=1024):
    samples = np.random.uniform(min, max, size=l)
    return [int(round(x)) for x in samples] # Probably don't need to round but whatever

def profile(batch_size=256, num_req=10000):
    # ===============================
    # Build request list
    # ===============================
    engine = Engine()
    base_prompt = """
    This is a long random prompt that is used to test the scheduler,
    it is definately longer than the maximum prefill length or input
    length depending on what you want to call. It is also a very long
    prompt that is used to test the scheduler, it is definately longer
    than the maximum prefill length or input length depending on what
    you want to call. There are also a lot of tokens in this prompt,
    so it is a good test of the scheduler. Going to the next line.
    """ * 10
    tokenized_prompt = engine.tokenizer.encode(base_prompt, return_tensors="pt")[0]
    requests = []
    input_lengths = sample_uniform_ints(num_req)
    output_lengths = sample_uniform_ints(num_req)
    for i in range(num_req):
        requests.append(InputRequest(
            engine.tokenizer.decode(tokenized_prompt[:input_lengths[i]]),
            input_lengths[i] + output_lengths[i]
        ))

    del engine
    gc.collect()
    torch.cuda.empty_cache()
    # ===============================
    # Naive scheduler (256 at a time)
    # ===============================
    engine_naive = Engine()
    scheduler_naive = Scheduler(engine_naive, req_batch_size=batch_size)

    torch.cuda.synchronize()
    naive_queue_times = []
    naive_iteration_times = []
    for i in tqdm(range(0, num_req, batch_size)):
        # Time queueing for this batch
        queue_start, queue_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        queue_start.record()
        for req in requests[i:i+batch_size]:
            scheduler_naive.add_req(req)
        queue_end.record()
        queue_end.synchronize()
        naive_queue_times.append(queue_start.elapsed_time(queue_end))
        # Time iterations for this batch
        while not scheduler_naive.finished():
            iter_start, iter_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            iter_start.record()
            scheduler_naive.run() # finish all requests in batch
            iter_end.record()
            iter_end.synchronize()
            naive_iteration_times.append(iter_start.elapsed_time(iter_end))
        gc.collect()
        torch.cuda.empty_cache()
    naive_total_time = sum(naive_queue_times) + sum(naive_iteration_times)
    print(f"Naive scheduler total time: {naive_total_time} ms (queue: {sum(naive_queue_times)} ms, iterations: {sum(naive_iteration_times)} ms)")
    np.save("results/naive_times.npy", np.array(naive_iteration_times))

    # Clean up
    del engine_naive
    del scheduler_naive
    gc.collect()
    torch.cuda.empty_cache()
    # ===============================
    # Continous scheduler (continuous batching)
    # ===============================
    engine_continous = Engine()
    scheduler_continous = Scheduler(engine_continous, req_batch_size=batch_size)

    torch.cuda.synchronize()
    # Time queueing
    queue_start, queue_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    queue_start.record()
    for req in requests:
        scheduler_continous.add_req(req)
    queue_end.record()
    queue_end.synchronize()
    queue_time = queue_start.elapsed_time(queue_end)
    # Per-iteration timing
    iteration_times = []
    while not scheduler_continous.finished():
        iter_start, iter_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        iter_start.record()
        scheduler_continous.run()
        iter_end.record()
        iter_end.synchronize()
        iteration_times.append(iter_start.elapsed_time(iter_end))
    total_time = queue_time + sum(iteration_times)
    print(f"Continous scheduler total time: {total_time} ms (queue: {queue_time} ms, iterations: {sum(iteration_times)} ms)")
    np.save("results/continous_times.npy", np.array(iteration_times))

    # Clean up
    del engine_continous
    del scheduler_continous
    gc.collect()
    torch.cuda.empty_cache()

# example_run()
profile()
