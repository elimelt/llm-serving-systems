from chunked_engine import Engine
from chunked_scheduler import Scheduler, InputRequest
import numpy as np
import gc
import torch

def sample_lognormal_ints(l, mean=6.0, sigma=0.7):
    samples = np.random.lognormal(mean, sigma, size=l)
    return [int(round(x)) for x in samples]

def sample_uniform_ints(l, min=1, max=1024):
    samples = np.random.uniform(min, max, size=l)
    return [int(round(x)) for x in samples] # Probably don't need to round but whatever

def example_run():
    engine = Engine()
    scheduler = Scheduler(engine, token_batch_size=1024)

    sample_prompts = ["Today is a rainy day"] * 1024 + ["UW is"] * 1024

    # Enqueue and run
    for prompt in sample_prompts:
        scheduler.add_req(InputRequest(prompt, output_len=100))
        # scheduler.run()

    # Drain remaining requests
    while not scheduler.finished():
        scheduler.run()

    # scheduler.print_completed()

def profile(token_batch_size=1024, num_req=10000):
    # num_req = 5
    engine = Engine()
    scheduler_continous = Scheduler(engine, token_batch_size=token_batch_size)
    # ===============================
    # Build request list
    # ===============================
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
    input_lengths = sample_lognormal_ints(num_req)
    output_lengths = sample_uniform_ints(num_req)
    for i in range(num_req):
        requests.append(InputRequest(
            engine.tokenizer.decode(tokenized_prompt[:input_lengths[i]]),
            input_lengths[i] + output_lengths[i]
        ))

    # ===============================
    # Profile chunked scheduler
    # ===============================
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
    print(f"Chunked scheduler total time: {total_time} ms (queue: {queue_time} ms, iterations: {sum(iteration_times)} ms)")
    np.save("results/chunked_times.npy", np.array(iteration_times))

    # Clean up
    del engine
    del scheduler_continous
    gc.collect()
    torch.cuda.empty_cache()

# example_run()
profile()
