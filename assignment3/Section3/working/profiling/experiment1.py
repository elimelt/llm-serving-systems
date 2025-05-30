import gc
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from profiling.engine import ProfiledEngine, log2_range
from profiling.utils import gpu_warmup, write_experiment_csv
from flashinfer_pipeline import Request

BATCH_SIZE = 128
PREFILL_LEN = 1024
DECODE_LENS = log2_range(5, 10)
PROMPT = "Hello world! " * (PREFILL_LEN // 3)


def experiment1():
    print("\n[Experiment 1] Batch size 128, prefill 1024, decode length 2^5-2^10")
    prefill_times = []
    decode_times = []
    for decode_len in DECODE_LENS:
        print(f"Decoding length: {decode_len}")
        engine = ProfiledEngine()
        gpu_warmup(engine, batch_size=BATCH_SIZE, prefill_len=PREFILL_LEN, decode_len=decode_len)
        
        requests = []
        for i in range(BATCH_SIZE):
            prompt_ids = engine.tokenizer(PROMPT[:PREFILL_LEN], return_tensors="pt").input_ids[0][:PREFILL_LEN]
            requests.append(Request(i, prompt_ids, decode_len))
        
        # Prefill timing
        torch.cuda.synchronize()
        prefill_outputs, timings_prefill = engine.run(requests, num_decode_req=0)
        prefill_times.append(timings_prefill.copy())

        for i in range(len(requests)):
            new_tok = prefill_outputs[i].unsqueeze(0)
            requests[i].output_token_ids = torch.cat(
                [requests[i].output_token_ids, new_tok], dim=0
            )

        # Decode timing
        decode_timings = []
        for _ in range(decode_len):
            torch.cuda.synchronize()
            decode_outputs, timings_decode = engine.run(requests, num_decode_req=len(requests))
            decode_timings.append(timings_decode.copy())

            for i in range(len(requests)):
                new_tok = decode_outputs[i].unsqueeze(0)
                requests[i].output_token_ids = torch.cat(
                    [requests[i].output_token_ids, new_tok], dim=0
                )

        decode_times.append(decode_timings.copy())
        del engine
        del requests
        torch.cuda.empty_cache()
        gc.collect()
    # --- Stack timings for plotting and write CSV ---
    os.makedirs("profile_results", exist_ok=True)

    # Aggregate prefill timings (just use the first timings dict for keys)
    fine_keys = list(prefill_times[0].keys())
    # For decode, sum over all decode steps for each run
    decode_fine_keys = list(decode_times[0][0].keys()) if decode_times[0] else []

    # Stack decode timings: sum each key over all decode steps for each run
    decode_breakdown = {k: [] for k in decode_fine_keys}
    last_decode_breakdown = {k: [] for k in decode_fine_keys}
    for decode_run in decode_times:
        for k in decode_fine_keys:
            decode_breakdown[k].append(sum(step[k] for step in decode_run))
            last_decode_breakdown[k].append(decode_run[-1][k])

    # Main timing keys for plotting
    timing_keys = ["prefill_time", "decode_time", "total_time"]
    param_keys = ["decode_len"]
    # Compose runs for write_experiment_csv
    runs = []
    for i, decode_len in enumerate(DECODE_LENS):
        prefill_time = prefill_times[i]["total"]
        decode_time = sum(step["total"] for step in decode_times[i])
        total_time = prefill_time + decode_time
        run = {"decode_len": decode_len, "prefill_time": prefill_time, "decode_time": decode_time, "total_time": total_time}
        # Add decode breakdowns (sum over all decode steps)
        for k in decode_fine_keys:
            run[f"decode_{k}"] = decode_breakdown[k][i]
        # Add last decode step breakdowns
        for k in decode_fine_keys:
            run[f"last_decode_{k}"] = last_decode_breakdown[k][i]
        # Add fine-grained prefill keys
        for k in fine_keys:
            run[f"prefill_{k}"] = prefill_times[i][k]
        runs.append(run)

    # Write CSV
    decode_breakdown_keys = [f"decode_{k}" for k in decode_fine_keys]
    last_decode_breakdown_keys = [f"last_decode_{k}" for k in decode_fine_keys]
    prefill_fine_keys = [f"prefill_{k}" for k in fine_keys]
    write_experiment_csv(
        "profile_results/exp1.csv",
        runs,
        param_keys,
        timing_keys,
        decode_breakdown_keys + last_decode_breakdown_keys + prefill_fine_keys
    )
    print("[exp1] Results saved to profile_results/exp1.csv (summary + fine-grained)")


if __name__ == "__main__":
    experiment1() 