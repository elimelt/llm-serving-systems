import torch
import random
from typing import List, Union
from chunked_engine import Engine as ChunkedEngine
from chunked_scheduler import Scheduler as ChunkedScheduler, InputRequest as ChunkedInputRequest
from continous_engine import Engine as ContinousEngine
from continous_scheduler import Scheduler as ContinousScheduler, InputRequest as ContinousInputRequest
import sys
import os
import gc
# Ensure assignment2 is in sys.path for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../assignment2')))
from no_kv import Engine as NoKVEngine

DEBUG = False

def generate_prompt_outputlen_inputids(engine, num_requests, max_input_len, max_output_len):
    random.seed(42)
    torch.manual_seed(42)
    requests = []
    for _ in range(num_requests):
        input_len = random.randint(1, max_input_len)
        raw = torch.randint(0, engine.tokenizer.vocab_size, (input_len,))
        prompt = engine.tokenizer.decode(raw.tolist(), skip_special_tokens=True)
        input_ids = engine.tokenizer.encode(prompt, return_tensors="pt")[0]
        output_len = random.randint(1, max_output_len) + len(input_ids)
        requests.append((prompt, output_len, input_ids.tolist()))
    return requests

def run_scheduler_collect_outputs(scheduler, input_requests, decode_fn):
    # Add all requests
    for req in input_requests:
        scheduler.add_req(req)
    # Run until finished
    while not scheduler.finished():
        scheduler.run()
    # Collect outputs (decoded text)
    outputs = {}
    for req in scheduler.completed:
        outputs[req.request_id] = req.output_token_ids
    return outputs

def main(num_requests=10, max_input_len=30, max_output_len=30):
    chunked_engine = ChunkedEngine()
    chunked_scheduler = ChunkedScheduler(chunked_engine, token_batch_size=32)
    # Generate random prompts and output lengths using helper
    print(f"Generating {num_requests} random requests...")
    input_requests = generate_prompt_outputlen_inputids(chunked_engine, num_requests, max_input_len, max_output_len)
    # Run chunked
    print("Running chunked engine...")
    chunked_inputs = [ChunkedInputRequest(prompt, out_len) for prompt, out_len, _ in input_requests]
    chunked_outputs = run_scheduler_collect_outputs(chunked_scheduler, chunked_inputs, chunked_engine.tokenizer.decode)
    # Clean up
    del chunked_scheduler
    del chunked_engine
    gc.collect()
    torch.cuda.empty_cache()

    continous_engine = ContinousEngine()
    continous_scheduler = ContinousScheduler(continous_engine, req_batch_size=32)
    # Run continous
    print("Running continous engine...")
    continous_inputs = [ContinousInputRequest(prompt, out_len) for prompt, out_len, _ in input_requests]
    continous_outputs = run_scheduler_collect_outputs(continous_scheduler, continous_inputs, continous_engine.tokenizer.decode)
    # Compare outputs with accuracy tracking
    print("Comparing chunked and continous outputs...")
    match_count = 0
    mismatch_count = 0
    for i in range(num_requests):
        c_out = chunked_outputs[i]
        t_out = continous_outputs[i]
        if torch.equal(c_out, t_out):
            match_count += 1
        else:
            mismatch_count += 1
            if DEBUG:
                print(f"Mismatch for request {i} (prompt: '{input_requests[i][0]}'):")
                print(f"Chunked: '{continous_engine.tokenizer.decode(c_out.tolist(), skip_special_tokens=True)}'")
                print(f"\t {c_out.tolist()}")
                print(f"Continous: '{continous_engine.tokenizer.decode(t_out.tolist(), skip_special_tokens=True)}'")
                print(f"\t {t_out.tolist()}")
                print("-" * 100)
    accuracy = match_count / num_requests * 100
    print(f"Chunked vs Continous accuracy: {accuracy:.2f}% ({match_count}/{num_requests} matched, {mismatch_count} mismatches)")
    # Clean up
    del continous_scheduler
    del continous_engine
    gc.collect()
    torch.cuda.empty_cache()
    print("=" * 100)
    print("Running no_kv engine for all requests...")
    print("=" * 100)

    # Run no_kv
    no_kv_engine = NoKVEngine()
    match_count = 0
    mismatch_count = 0
    for i, (prompt, out_len, input_ids) in enumerate(input_requests):
        # Generate using no_kv, matching the number of tokens generated
        output_ids = input_ids.copy()
        while len(output_ids) < out_len:
            new_token = no_kv_engine.run(output_ids, prefill=True)
            output_ids.append(new_token)
        c_out = chunked_outputs[i]
        if torch.equal(c_out, torch.tensor(output_ids)):
            match_count += 1
        else:
            mismatch_count += 1
            if DEBUG:
                print(f"Mismatch for request {i} (prompt: '{prompt}' out_len: {out_len}) with no_kv:")
                print(f"Chunked: '{no_kv_engine.tokenizer.decode(c_out.tolist(), skip_special_tokens=True)}'")
                print(f"\t {c_out.tolist()}")
                print(f"no_kv: '{no_kv_engine.tokenizer.decode(output_ids, skip_special_tokens=True)}'")
                print(f"\t {output_ids}")
                print("-" * 100)
    accuracy = match_count / num_requests * 100
    print(f"Chunked vs no_kv accuracy: {accuracy:.2f}% ({match_count}/{num_requests} matched, {mismatch_count} mismatches)")

if __name__ == "__main__":
    main(num_requests=512, max_input_len=256, max_output_len=256)