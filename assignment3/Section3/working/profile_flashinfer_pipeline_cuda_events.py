import gc
import os
import csv
import flashinfer
import numpy as np
import torch
from flashinfer_pipeline import DistKVCache, Engine, Request, build_kv_metadata

# Output directories
os.makedirs("profile_results_cuda_events", exist_ok=True)

def synchronize():
    torch.cuda.synchronize()

def log2_range(start, stop):
    return [2 ** i for i in range(start, stop + 1)]

# --- Helper for CUDA event timing ---
class CUDATimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        
    def start(self):
        self.start_event.record()
        
    def end(self):
        self.end_event.record()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event) / 1000.0  # Convert ms to s

# --- Fine-grained profiling via monkey-patching ---
class ProfiledEngine(Engine):
    def __init__(self):
        super().__init__()
        self.last_embedding_time = 0.0
        self.last_layer_times = []  # List of dicts per layer
        self.last_logits_time = 0.0
        self.last_total_time = 0.0
        self.last_finegrained_times = {}
        self.timer = CUDATimer()

    def run(self, requests, num_decode_req=0):
        self.last_layer_times = []
        self.last_finegrained_times = {}
        
        total_timer = CUDATimer()
        total_timer.start()
        
        with torch.inference_mode():
            # --- Input tensor construction ---
            input_timer = CUDATimer()
            input_timer.start()
            pieces = []
            indptr = [0]
            for idx, req in enumerate(requests):
                if idx < num_decode_req:
                    pieces.append(req.output_token_ids[-1:])
                    indptr.append(indptr[-1] + 1)
                else:
                    pieces.append(req.prompt_token_ids)
                    indptr.append(indptr[-1] + req.prompt_length)
            input_tensor = torch.cat(pieces).to("cuda")
            indptr_tensor = torch.tensor(indptr, dtype=torch.int32, device="cuda")
            self.last_finegrained_times['input_tensor'] = input_timer.end()

            # --- KV cache allocation ---
            kv_alloc_timer = CUDATimer()
            kv_alloc_timer.start()
            for idx, r in enumerate(requests):
                if r.request_id not in self.kv_cache_map and idx >= num_decode_req:
                    self.kv_cache_map[r.request_id] = DistKVCache(self.pool)
            seq_lens_before = [self.kv_cache_map[r.request_id].seqlen for r in requests]
            seq_lens_before_t = torch.tensor(seq_lens_before, dtype=torch.int32, device="cuda")
            for idx, r in enumerate(requests):
                if idx >= num_decode_req:
                    self.kv_cache_map[r.request_id].allocate_tokens(r.prompt_length)
                elif self.kv_cache_map[r.request_id].seqlen < r.current_length:
                    self.kv_cache_map[r.request_id].allocate_tokens(1)
            seq_lens_after = [self.kv_cache_map[r.request_id].seqlen for r in requests]
            seq_lens_after_t = torch.tensor(seq_lens_after, dtype=torch.int32, device="cuda")
            self.last_finegrained_times['kv_cache_alloc'] = kv_alloc_timer.end()

            # --- Metadata building ---
            meta_timer = CUDATimer()
            meta_timer.start()
            kv_indptr, kv_indices, kv_last_page_len = build_kv_metadata(
                [self.kv_cache_map[r.request_id] for r in requests]
            )
            self.last_finegrained_times['kv_metadata'] = meta_timer.end()

            # --- Planning (prefill/decoder) ---
            plan_timer = CUDATimer()
            plan_timer.start()
            if not len(requests) - num_decode_req == 0:
                self.prefill_wrapper.plan(
                    indptr_tensor, kv_indptr, kv_indices, kv_last_page_len,
                    self.num_qo_heads, self.num_kv_heads, self.head_dim, self.page_size, causal=True
                )
            if num_decode_req > 0:
                self.decode_wrapper.plan(
                    kv_indptr, kv_indices, kv_last_page_len,
                    self.num_qo_heads, self.num_kv_heads, self.head_dim, self.page_size
                )
            self.last_finegrained_times['planning'] = plan_timer.end()

            # --- Embedding ---
            emb_timer = CUDATimer()
            emb_timer.start()
            hidden = self.weights["embedding"][input_tensor]
            self.last_embedding_time = emb_timer.end()
            self.last_finegrained_times['embedding'] = self.last_embedding_time

            # --- Per-layer forward ---
            layer_times = []
            for layer in range(self.layers):
                layer_timer = CUDATimer()
                layer_timer.start()
                
                # --- LayerNorm before attention ---
                ln_attn_timer = CUDATimer()
                ln_attn_timer.start()
                rms = torch.sqrt(hidden.square().mean(-1, keepdim=True) + 1e-5)
                ln_attn_in = (hidden / rms).to(torch.float16) * self.weights["layernormAttn_weight"][layer]
                ln_attn_time = ln_attn_timer.end()
                
                # --- QKV projections ---
                qkv_timer = CUDATimer()
                qkv_timer.start()
                k = ln_attn_in.matmul(self.weights["self_attn_k_proj_weight"][layer].T).view(-1, self.num_kv_heads, self.head_dim)
                v = ln_attn_in.matmul(self.weights["self_attn_v_proj_weight"][layer].T).view(-1, self.num_kv_heads, self.head_dim)
                q = ln_attn_in.matmul(self.weights["self_attn_q_proj_weight"][layer].T).view(-1, self.num_qo_heads, self.head_dim)
                qkv_time = qkv_timer.end()
                
                # --- RoPE ---
                rope_timer = CUDATimer()
                rope_timer.start()
                flashinfer.rope.apply_rope_inplace(q, k, indptr_tensor, seq_lens_before_t, rope_theta=500_000.0)
                rope_time = rope_timer.end()
                
                # --- KV append ---
                append_timer = CUDATimer()
                append_timer.start()
                batch_indices, positions = flashinfer.get_batch_indices_positions(
                    indptr_tensor, seq_lens_after_t, k.shape[0]
                )
                flashinfer.append_paged_kv_cache(
                    k, v, batch_indices, positions, self.pool.get_paged_kv_cache(layer),
                    kv_indices, kv_indptr, kv_last_page_len, kv_layout="HND"
                )
                append_time = append_timer.end()
                
                # --- Attention kernel ---
                attn_out_prefill = attn_out_decode = None
                attn_kernel_timer = CUDATimer()
                attn_kernel_timer.start()
                if not len(requests) - num_decode_req == 0:
                    attn_out_prefill = self.prefill_wrapper.run(q, self.pool.get_paged_kv_cache(layer))
                if num_decode_req > 0:
                    attn_out_decode = self.decode_wrapper.run(q, self.pool.get_paged_kv_cache(layer))
                if attn_out_prefill is not None and attn_out_decode is not None:
                    attn_out = torch.cat((attn_out_decode, attn_out_prefill), dim=0)
                elif attn_out_prefill is not None:
                    attn_out = attn_out_prefill
                elif attn_out_decode is not None:
                    attn_out = attn_out_decode
                attn_out = attn_out.reshape(attn_out.shape[0], -1)
                attn_kernel_time = attn_kernel_timer.end()
                
                # --- O projection ---
                o_proj_timer = CUDATimer()
                o_proj_timer.start()
                hidden = attn_out.matmul(self.weights["o_proj_weight"][layer].T) + hidden
                o_proj_time = o_proj_timer.end()
                
                # --- LayerNorm before FFN ---
                ln_ffn_timer = CUDATimer()
                ln_ffn_timer.start()
                rms = torch.sqrt(hidden.square().mean(-1, keepdim=True) + 1e-5)
                ln_ffn_in = (hidden / rms).to(torch.float16) * self.weights["layernormFFN_weight"][layer]
                ln_ffn_time = ln_ffn_timer.end()
                
                # --- FFN projections and activation ---
                ffn_timer = CUDATimer()
                ffn_timer.start()
                up = ln_ffn_in.matmul(self.weights["up_proj_weight"][layer].T)
                gate = ln_ffn_in.matmul(self.weights["gate_proj_weight"][layer].T)
                ffn_out = (up * torch.nn.functional.silu(gate)).matmul(self.weights["down_proj_weight"][layer].T)
                hidden = ffn_out + hidden
                ffn_time = ffn_timer.end()
                
                # Record total layer time
                layer_total_time = layer_timer.end()
                
                # --- Store timings for this layer ---
                layer_times.append({
                    'ln_attn': ln_attn_time,
                    'qkv_proj': qkv_time,
                    'rope': rope_time,
                    'kv_append': append_time,
                    'attn_kernel': attn_kernel_time,
                    'o_proj': o_proj_time,
                    'ln_ffn': ln_ffn_time,
                    'ffn': ffn_time,
                    'total': layer_total_time
                })
            self.last_layer_times = layer_times
            self.last_finegrained_times['layers'] = sum([l['total'] for l in layer_times])

            # --- Logits ---
            logits_timer = CUDATimer()
            logits_timer.start()
            rms = torch.sqrt(hidden.square().mean(-1, keepdim=True) + 1e-5)
            logits = (hidden / rms).to(torch.float16) * self.weights["model_layernorm_weight"]
            logits = logits.matmul(self.weights["lm_head_weight"].T)
            sample_ids = torch.argmax(logits, dim=-1)
            last_token_indices = (indptr_tensor[1:] - 1).long()
            self.last_logits_time = logits_timer.end()
            self.last_finegrained_times['logits'] = self.last_logits_time
            
        self.last_total_time = total_timer.end()
        self.last_finegrained_times['total'] = self.last_total_time
        return sample_ids[last_token_indices].cpu(), self.last_total_time

# --- GPU warmup utility ---
def gpu_warmup(engine, batch_size=8, prefill_len=64, decode_len=8):
    prompt = "Hello world! " * (prefill_len // 3)
    prompts = [prompt[:prefill_len] for _ in range(batch_size)]
    requests = []
    for i in range(batch_size):
        prompt_ids = engine.tokenizer(prompts[i], return_tensors="pt").input_ids[0][:prefill_len]
        requests.append(Request(i, prompt_ids, decode_len))
    synchronize()
    engine.run(requests, num_decode_req=0)
    for i in range(len(requests)):
        requests[i].output_token_ids = torch.cat([
            requests[i].output_token_ids,
            torch.zeros(1, dtype=requests[i].output_token_ids.dtype, device=requests[i].output_token_ids.device)
        ], dim=0)
    for _ in range(decode_len):
        engine.run(requests, num_decode_req=len(requests))
        for i in range(len(requests)):
            requests[i].output_token_ids = torch.cat([
                requests[i].output_token_ids,
                torch.zeros(1, dtype=requests[i].output_token_ids.dtype, device=requests[i].output_token_ids.device)
            ], dim=0)
    synchronize()

# --- Helper for stacking timings ---
def stack_layer_times(layer_times_list, key):
    # Sum the timing for a given key across all layers in a run
    return [sum(layer[key] for layer in layers) for layers in layer_times_list]

def stack_layer_times_all(layer_times_list):
    # Return a dict of lists for all keys
    keys = layer_times_list[0][0].keys()
    return {k: [sum(layer[k] for layer in layers) for layers in layer_times_list] for k in keys}

# --- Experiment 1: Batch size 128, prefill 1024, decode length 2^5-2^10 ---
def experiment1():
    print("\n[Experiment 1] Batch size 128, prefill 1024, decode length 2^5-2^10")
    batch_size = 128
    prefill_len = 1024
    decode_lens = log2_range(5, 10)  # 32, 64, ..., 1024
    prefill_times = []
    decode_times = []
    total_times = []
    layer_times_decode = []
    layer_times_prefill = []
    embedding_times = []
    logits_times = []
    finegrained_prefill = []
    finegrained_decode = []
    
    # CUDA event timer for overall timing
    timer = CUDATimer()
    
    for decode_len in decode_lens:
        engine = ProfiledEngine()
        gpu_warmup(engine, batch_size=batch_size, prefill_len=prefill_len, decode_len=decode_len)
        prompt = "Hello world! " * (prefill_len // 3)
        prompts = [prompt[:prefill_len] for _ in range(batch_size)]
        requests = []
        for i in range(batch_size):
            prompt_ids = engine.tokenizer(prompts[i], return_tensors="pt").input_ids[0][:prefill_len]
            requests.append(Request(i, prompt_ids, decode_len))
        
        # Prefill timing with CUDA events
        synchronize()
        timer.start()
        _, _ = engine.run(requests, num_decode_req=0)
        prefill_time = timer.end()
        
        prefill_times.append(prefill_time)
        embedding_times.append(engine.last_embedding_time)
        layer_times_prefill.append(engine.last_layer_times.copy())
        logits_times.append(engine.last_logits_time)
        finegrained_prefill.append(engine.last_finegrained_times.copy())
        
        for i in range(len(requests)):
            requests[i].output_token_ids = torch.cat([
                requests[i].output_token_ids,
                torch.zeros(1, dtype=requests[i].output_token_ids.dtype, device=requests[i].output_token_ids.device)
            ], dim=0)
            
        decode_time = 0.0
        decode_layer_times = []
        decode_finegrained = []
        
        for _ in range(decode_len):
            synchronize()
            timer.start()
            _, _ = engine.run(requests, num_decode_req=len(requests))
            step_time = timer.end()
            
            decode_time += step_time
            decode_layer_times.append(engine.last_layer_times.copy())
            decode_finegrained.append(engine.last_finegrained_times.copy())
            
            for i in range(len(requests)):
                requests[i].output_token_ids = torch.cat([
                    requests[i].output_token_ids,
                    torch.zeros(1, dtype=requests[i].output_token_ids.dtype, device=requests[i].output_token_ids.device)
                ], dim=0)
                
        total_time = prefill_time + decode_time
        total_times.append(total_time)
        decode_times.append(decode_time)
        layer_times_decode.append([layer for step in decode_layer_times for layer in step])
        finegrained_decode.append(decode_finegrained)
        
        del engine
        torch.cuda.empty_cache()
        gc.collect()
        
    # --- Stack timings for plotting ---
    prefill_breakdown = stack_layer_times_all(layer_times_prefill)
    decode_breakdown = stack_layer_times_all(layer_times_decode)
    prefill_breakdown['embedding'] = embedding_times
    prefill_breakdown['logits'] = logits_times
    decode_breakdown['embedding'] = [0]*len(decode_lens)
    decode_breakdown['logits'] = [0]*len(decode_lens)
    
    # Save CSV (summary + finegrained breakdown)
    fine_keys = list(finegrained_prefill[0].keys())
    decode_fine_keys = list(finegrained_decode[0][0].keys()) if finegrained_decode[0] else []
    decode_keys = [k for k in decode_breakdown if not all(v == 0 for v in decode_breakdown[k])]
    prefill_keys = list(prefill_breakdown.keys())
    
    with open("profile_results_cuda_events/exp1.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["decode_len", "prefill_time", "decode_time", "total_time"] \
                 + [f"prefill_{k}" for k in prefill_keys] \
                 + [f"decode_{k}" for k in decode_keys] \
                 + [f"prefill_fine_{k}" for k in fine_keys] \
                 + [f"decode_fine_{k}" for k in decode_fine_keys]
        writer.writerow(header)
        for i, d in enumerate(decode_lens):
            row = [d, prefill_times[i], decode_times[i], total_times[i]]
            row += [prefill_breakdown[k][i] for k in prefill_keys]
            row += [decode_breakdown[k][i] for k in decode_keys]
            row += [finegrained_prefill[i][k] for k in fine_keys]
            # For decode, sum over all decode steps
            decode_sums = {k: 0.0 for k in decode_fine_keys}
            for step in finegrained_decode[i]:
                for k in decode_fine_keys:
                    decode_sums[k] += step[k]
            row += [decode_sums[k] for k in decode_fine_keys]
            writer.writerow(row)
    print("[exp1] Results saved to profile_results_cuda_events/exp1.csv (summary + fine-grained)")

# --- Experiment 2: Batch size 1, prefill 2^8-2^16, profile prefill time ---
def experiment2():
    print("\n[Experiment 2] Batch size 1, prefill 2^8-2^16, profile prefill time")
    batch_size = 1
    prefill_lens = log2_range(8, 16)
    decode_len = 1
    prefill_times = []
    layer_times_prefill = []
    embedding_times = []
    logits_times = []
    finegrained_prefill = []
    
    # CUDA event timer for overall timing
    timer = CUDATimer()
    
    for prefill_len in prefill_lens:
        engine = ProfiledEngine()
        gpu_warmup(engine, batch_size=batch_size, prefill_len=prefill_len, decode_len=decode_len)
        prompt = "Hello world! " * (prefill_len // 3)
        prompt_ids = engine.tokenizer(prompt[:prefill_len], return_tensors="pt").input_ids[0][:prefill_len]
        requests = [Request(0, prompt_ids, decode_len)]
        
        synchronize()
        timer.start()
        _, _ = engine.run(requests, num_decode_req=0)
        prefill_time = timer.end()
        
        prefill_times.append(prefill_time)
        embedding_times.append(engine.last_embedding_time)
        layer_times_prefill.append(engine.last_layer_times.copy())
        logits_times.append(engine.last_logits_time)
        finegrained_prefill.append(engine.last_finegrained_times.copy())
        
        del engine
        torch.cuda.empty_cache()
        gc.collect()
        
    prefill_breakdown = stack_layer_times_all(layer_times_prefill)
    prefill_breakdown['embedding'] = embedding_times
    prefill_breakdown['logits'] = logits_times
    fine_keys = list(finegrained_prefill[0].keys())
    prefill_keys = list(prefill_breakdown.keys())
    
    with open("profile_results_cuda_events/exp2.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["prefill_len", "prefill_time"] + [f"prefill_{k}" for k in prefill_keys] + [f"prefill_fine_{k}" for k in fine_keys]
        writer.writerow(header)
        for i, p in enumerate(prefill_lens):
            row = [p, prefill_times[i]]
            row += [prefill_breakdown[k][i] for k in prefill_keys]
            row += [finegrained_prefill[i][k] for k in fine_keys]
            writer.writerow(row)
    print("[exp2] Results saved to profile_results_cuda_events/exp2.csv (summary + fine-grained)")

# --- Experiment 3: Batch size 2^0-2^10, prefill 128, decode 128 ---
def experiment3():
    print("\n[Experiment 3] Batch size 2^0-2^10, prefill 128, decode 128")
    batch_sizes = log2_range(0, 10)
    prefill_len = 128
    decode_len = 128
    throughput = []
    total_times = []
    prefill_times = []
    decode_times = []
    layer_times_prefill = []
    embedding_times = []
    logits_times = []
    finegrained_prefill = []
    
    # CUDA event timer for overall timing
    timer = CUDATimer()
    
    for batch_size in batch_sizes:
        engine = ProfiledEngine()
        gpu_warmup(engine, batch_size=batch_size, prefill_len=prefill_len, decode_len=decode_len)
        prompt = "Hello world! " * (prefill_len // 3)
        prompts = [prompt[:prefill_len] for _ in range(batch_size)]
        requests = []
        for i in range(batch_size):
            prompt_ids = engine.tokenizer(prompts[i], return_tensors="pt").input_ids[0][:prefill_len]
            requests.append(Request(i, prompt_ids, decode_len))
        
        synchronize()
        timer.start()
        _, _ = engine.run(requests, num_decode_req=0)
        prefill_time = timer.end()
        
        prefill_times.append(prefill_time)
        embedding_times.append(engine.last_embedding_time)
        layer_times_prefill.append(engine.last_layer_times.copy())
        logits_times.append(engine.last_logits_time)
        finegrained_prefill.append(engine.last_finegrained_times.copy())
        
        for i in range(len(requests)):
            requests[i].output_token_ids = torch.cat([
                requests[i].output_token_ids,
                torch.zeros(1, dtype=requests[i].output_token_ids.dtype, device=requests[i].output_token_ids.device)
            ], dim=0)
            
        decode_time = 0.0
        for _ in range(decode_len):
            synchronize()
            timer.start()
            _, _ = engine.run(requests, num_decode_req=len(requests))
            step_time = timer.end()
            
            decode_time += step_time
            for i in range(len(requests)):
                requests[i].output_token_ids = torch.cat([
                    requests[i].output_token_ids,
                    torch.zeros(1, dtype=requests[i].output_token_ids.dtype, device=requests[i].output_token_ids.device)
                ], dim=0)
                
        total = prefill_time + decode_time
        total_times.append(total)
        decode_times.append(decode_time)
        tokens = batch_size * (prefill_len + decode_len)
        throughput.append(tokens / total if total > 0 else 0)
        
        del engine
        torch.cuda.empty_cache()
        gc.collect()
        
    prefill_breakdown = stack_layer_times_all(layer_times_prefill)
    prefill_breakdown['embedding'] = embedding_times
    prefill_breakdown['logits'] = logits_times
    fine_keys = list(finegrained_prefill[0].keys())
    prefill_keys = list(prefill_breakdown.keys())
    
    with open("profile_results_cuda_events/exp3.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["batch_size", "prefill_time", "decode_time", "total_time", "throughput"] + [f"prefill_{k}" for k in prefill_keys] + [f"prefill_fine_{k}" for k in fine_keys]
        writer.writerow(header)
        for i, b in enumerate(batch_sizes):
            row = [b, prefill_times[i], decode_times[i], total_times[i], throughput[i]]
            row += [prefill_breakdown[k][i] for k in prefill_keys]
            row += [finegrained_prefill[i][k] for k in fine_keys]
            writer.writerow(row)
    print("[exp3] Results saved to profile_results_cuda_events/exp3.csv (summary + fine-grained)")

if __name__ == "__main__":
    experiment1()
    experiment2()
    experiment3()
    print("\nAll experiments complete. See 'profile_results_cuda_events/' for outputs.")