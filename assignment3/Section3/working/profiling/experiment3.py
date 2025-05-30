import gc
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from profiling.engine import ProfiledEngine, log2_range
from profiling.utils import gpu_warmup, write_experiment_csv
from flashinfer_pipeline import Request, Engine

BATCH_SIZES = log2_range(0, 10)
PREFILL_LEN = 128
DECODE_LEN = 128
PROMPT = "Hello world! " * (PREFILL_LEN // 3)

def experiment3():
    print("\n[Experiment 3] Batch size 2^0-2^10, prefill 128, decode 128")
    total_times = []
    for batch_size in BATCH_SIZES:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        print(f"Batch size: {batch_size}")
        engine = Engine()
        gpu_warmup(engine, batch_size=batch_size, prefill_len=PREFILL_LEN, decode_len=DECODE_LEN)
        requests = []
        for i in range(batch_size):
            prompt_ids = engine.tokenizer(PROMPT[:PREFILL_LEN], return_tensors="pt").input_ids[0][:PREFILL_LEN]
            requests.append(Request(i, prompt_ids, DECODE_LEN))
        # Prefill timing
        torch.cuda.synchronize()
        start.record()
        prefill_outputs = engine.run(requests, num_decode_req=0)
        for i in range(len(requests)):
            new_tok = prefill_outputs[i].unsqueeze(0)
            requests[i].output_token_ids = torch.cat(
                [requests[i].output_token_ids, new_tok], dim=0
            )
        # Decode timing
        for _ in range(DECODE_LEN):
            decode_outputs = engine.run(requests, num_decode_req=len(requests))
            for i in range(len(requests)):
                new_tok = decode_outputs[i].unsqueeze(0)
                requests[i].output_token_ids = torch.cat(
                    [requests[i].output_token_ids, new_tok], dim=0
                )
        end.record()
        end.synchronize()
        total_time = start.elapsed_time(end) / 1000
        total_times.append(total_time)
        del engine
        del requests
        torch.cuda.empty_cache()
        gc.collect()
    # --- Stack timings for plotting and write CSV ---
    os.makedirs("profile_results", exist_ok=True)
    timing_keys = ["prefill_time", "decode_time", "total_time", "throughput"]
    param_keys = ["batch_size"]
    runs = []
    for i, batch_size in enumerate(BATCH_SIZES):
        total_time = total_times[i]
        tokens = batch_size * (PREFILL_LEN + DECODE_LEN)
        tput = tokens / total_time if total_time > 0 else 0
        run = {"batch_size": batch_size, "total_time": total_time, "throughput": tput}
        runs.append(run)
    write_experiment_csv(
        "profile_results/exp3.csv",
        runs,
        param_keys,
        timing_keys,
        []
    )
    print("[exp3] Results saved to profile_results/exp3.csv (summary only)")

if __name__ == "__main__":
    experiment3() 