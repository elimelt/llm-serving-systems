"""
Test script for transformer-tp2.py (Tensor Parallel) vs transformer-w3l1.py (Reference).

- Reference runs on a single GPU (the first in CUDA_VISIBLE_DEVICES).
- TP version runs on TP_WORLD_SIZE GPUs (each process uses one GPU).

Before running, set CUDA_VISIBLE_DEVICES to at least TP_WORLD_SIZE GPUs, e.g.:
    export CUDA_VISIBLE_DEVICES=0,1,2
    python assignment4/Section2/test_implementation.py
"""

TP_WORLD_SIZE = 2  # Set to 2 or 3 depending on your available GPUs

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import random
import sys
import os
from transformers import AutoTokenizer

# Add Section2 to sys.path for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def check_cuda_visible_devices():
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible is None:
        print("\033[93mWARNING: CUDA_VISIBLE_DEVICES is not set. All GPUs will be visible.\033[0m")
        return
    gpu_list = [x for x in cuda_visible.split(",") if x.strip()]
    if len(gpu_list) < TP_WORLD_SIZE:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES has {len(gpu_list)} GPUs, but TP_WORLD_SIZE={TP_WORLD_SIZE}.")

def generate_prompt_outputlen_inputids(tokenizer, num_requests, max_input_len, max_output_len):
    random.seed(42)
    torch.manual_seed(42)
    requests = []
    for _ in range(num_requests):
        input_len = random.randint(1, max_input_len)
        raw = torch.randint(0, tokenizer.vocab_size, (input_len,))
        prompt = tokenizer.decode(raw.tolist(), skip_special_tokens=True)
        input_ids = tokenizer.encode(prompt)
        output_len = random.randint(1, max_output_len) + len(input_ids)
        requests.append((prompt, output_len, input_ids))
    return requests

def run_tp(rank, world_size, input_ids, output_len, return_dict, port):
    import importlib
    import os
    try:
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        tp_impl = importlib.import_module("transformer-tp2")
        output_ids = input_ids.copy()
        for _ in range(output_len - len(output_ids)):
            new_token = tp_impl.run_one_iteration(output_ids, rank, world_size)
            output_ids.append(new_token)
        return_dict[rank] = output_ids
    except Exception as e:
        import traceback
        return_dict[rank] = f"ERROR: {e}\n{traceback.format_exc()}"

def main(num_requests=10, max_input_len=30, max_output_len=30):
    import importlib
    ref_impl = importlib.import_module("transformer-w3l1")
    check_cuda_visible_devices()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    input_requests = generate_prompt_outputlen_inputids(tokenizer, num_requests, max_input_len, max_output_len)
    match_count = 0
    mismatch_count = 0
    for i, (prompt, out_len, input_ids) in enumerate(input_requests):
        print(f"Request {i}: prompt='{prompt[:50]}...' (len={len(input_ids)}), out_len={out_len}")
        # Reference (single GPU)
        ref_output_ids = input_ids.copy()
        for _ in range(out_len - len(ref_output_ids)):
            new_token = ref_impl.run_one_iteration(ref_output_ids)
            ref_output_ids.append(new_token)
        # TP (TP_WORLD_SIZE GPUs)
        mp.set_start_method('spawn', force=True)
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []
        port = random.randint(20000, 40000)
        for rank in range(TP_WORLD_SIZE):
            p = mp.Process(target=run_tp, args=(rank, TP_WORLD_SIZE, input_ids, out_len, return_dict, port))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        tp_outputs = [return_dict.get(r, None) for r in range(TP_WORLD_SIZE)]
        if any(isinstance(out, str) and out.startswith("ERROR:") for out in tp_outputs):
            print(f"\033[91mError in TP worker(s):\033[0m")
            for r, out in enumerate(tp_outputs):
                if isinstance(out, str) and out.startswith("ERROR:"):
                    print(f"Rank {r} error:\n{out}")
            continue
        assert all(tp_out == tp_outputs[0] for tp_out in tp_outputs), "Mismatch between TP ranks!"
        tp_output_ids = tp_outputs[0]
        # Compare
        if ref_output_ids == tp_output_ids:
            match_count += 1
        else:
            mismatch_count += 1
            print(f"\033[91mMismatch for request {i}\033[0m")
            print(f"Prompt: {prompt}")
            print(f"\t{ref_output_ids}")
            print(f"TP: {tokenizer.decode(tp_output_ids, skip_special_tokens=True)}")
            print(f"\t{tp_output_ids}")
            print("-" * 80)
    accuracy = match_count / num_requests * 100
    print(f"TP vs Reference accuracy: {accuracy:.2f}% ({match_count}/{num_requests} matched, {mismatch_count} mismatches)")

if __name__ == "__main__":
    main(num_requests=128, max_input_len=128, max_output_len=128)
