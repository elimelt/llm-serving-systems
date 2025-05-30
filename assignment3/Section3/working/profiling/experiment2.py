import gc
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from profiling.engine import ProfiledEngine, log2_range
from profiling.utils import gpu_warmup, write_experiment_csv
from flashinfer_pipeline import Request

BATCH_SIZE = 1
PREFILL_LENS = log2_range(8, 16)
DECODE_LEN = 1
PROMPT = "Hello world! " * (2 ** 16 // 3)

def experiment2():
    print("\n[Experiment 2] Batch size 1, prefill 2^8-2^16, profile prefill time")
    prefill_times = []
    for prefill_len in PREFILL_LENS:
        print(f"Prefill length: {prefill_len}")
        engine = ProfiledEngine()
        gpu_warmup(engine, batch_size=BATCH_SIZE, prefill_len=prefill_len, decode_len=DECODE_LEN)
        prompt_ids = engine.tokenizer(PROMPT[:prefill_len], return_tensors="pt").input_ids[0][:prefill_len]
        requests = [Request(0, prompt_ids, DECODE_LEN)]
        # Prefill timing
        torch.cuda.synchronize()
        _, timings_prefill = engine.run(requests, num_decode_req=0)
        prefill_times.append(timings_prefill.copy())
        del engine
        del requests
        torch.cuda.empty_cache()
        gc.collect()
    # --- Stack timings for plotting and write CSV ---
    os.makedirs("profile_results", exist_ok=True)
    # Aggregate prefill timings (just use the first timings dict for keys)
    fine_keys = list(prefill_times[0].keys())
    # Main timing keys for plotting
    timing_keys = ["prefill_time"]
    param_keys = ["prefill_len"]
    # Compose runs for write_experiment_csv
    runs = []
    for i, prefill_len in enumerate(PREFILL_LENS):
        prefill_time = prefill_times[i]["total"]
        run = {"prefill_len": prefill_len, "prefill_time": prefill_time}
        # Add fine-grained prefill keys
        for k in fine_keys:
            run[f"prefill_{k}"] = prefill_times[i][k]
        runs.append(run)
    # Write CSV
    prefill_fine_keys = [f"prefill_{k}" for k in fine_keys]
    write_experiment_csv(
        "profile_results/exp2.csv",
        runs,
        param_keys,
        timing_keys,
        prefill_fine_keys
    )
    print("[exp2] Results saved to profile_results/exp2.csv (summary + fine-grained)")

if __name__ == "__main__":
    experiment2() 