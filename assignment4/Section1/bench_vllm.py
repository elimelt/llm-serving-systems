import torch
import vllm.benchmarks.throughput as vllm_bench
import argparse
from chunked_engine import Engine
from chunked_scheduler import Scheduler, InputRequest

def vllm():
    parser = argparse.ArgumentParser()
    vllm_bench.add_cli_args(parser)
    args = parser.parse_args()
    args.backend = "vllm"
    args.model = "/model/Meta-Llama-3-8B-Instruct"
    args.input_len = 512
    args.output_len = 512
    args.num_prompts = 10000
    args.output_json = "results/vllm_bench.json"
    vllm_bench.main(args=args)

def chunked(token_batch_size=1024, num_req=10000):
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
    """ * 6
    tokenized_prompt = engine.tokenizer.encode(base_prompt, return_tensors="pt")[0]
    requests = []
    input_lengths = 512
    output_lengths = 512
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

if __name__ == "__main__":
    # vllm()
    chunked()