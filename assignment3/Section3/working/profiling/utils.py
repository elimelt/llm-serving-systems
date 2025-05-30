import csv
import torch
from flashinfer_pipeline import Request

def gpu_warmup(engine, batch_size=8, prefill_len=64, decode_len=8):
    prompt = "Hello world! " * (prefill_len // 3)
    prompts = [prompt[:prefill_len] for _ in range(batch_size)]
    requests = []
    for i in range(batch_size):
        prompt_ids = engine.tokenizer(prompts[i], return_tensors="pt").input_ids[0][:prefill_len]
        requests.append(Request(i, prompt_ids, decode_len))
    torch.cuda.synchronize()
    # outputs, _ = engine.run(requests, num_decode_req=0)
    outputs = engine.run(requests, num_decode_req=0)
    for i in range(len(requests)):
        new_tok = outputs[i].unsqueeze(0)
        requests[i].output_token_ids = torch.cat([requests[i].output_token_ids, new_tok], dim=0)
    for _ in range(decode_len):
        # outputs, _ = engine.run(requests, num_decode_req=len(requests))
        outputs = engine.run(requests, num_decode_req=len(requests))
        for i in range(len(requests)):
            new_tok = outputs[i].unsqueeze(0)
            requests[i].output_token_ids = torch.cat([requests[i].output_token_ids, new_tok], dim=0)
    torch.cuda.synchronize()


def write_experiment_csv(filepath, runs, param_keys, timing_keys, fine_keys):
    """
    Write a unified CSV for all experiments.
    - runs: list of dicts, each dict contains all info for one run (params, timings, fine-grained, etc)
    - param_keys: list of parameter/experiment keys (e.g. batch_size, decode_len, prefill_len, throughput)
    - timing_keys: list of timing breakdown keys (e.g. prefill_time, decode_time, total_time, etc)
    - fine_keys: list of fine-grained timing keys (e.g. input_tensor, kv_cache_alloc, ...)
    """
    header = param_keys + timing_keys + fine_keys
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for run in runs:
            row = [run.get(k, "") for k in param_keys]
            row += [run.get(k, 0.0) for k in timing_keys]
            row += [run.get(k, 0.0) for k in fine_keys]
            writer.writerow(row) 