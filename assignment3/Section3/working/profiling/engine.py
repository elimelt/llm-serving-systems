import torch
import flashinfer
from flashinfer_pipeline import DistKVCache, Engine, build_kv_metadata

def log2_range(start, stop):
    return [2 ** i for i in range(start, stop + 1)] 

class CUDATimer:
    def __init__(self):
        self.events = {}  # Dictionary to store named events
        
    def start(self, name="default"):
        """Start timing an event with the given name"""
        if name not in self.events:
            self.events[name] = {
                'start': torch.cuda.Event(enable_timing=True),
                'end': torch.cuda.Event(enable_timing=True)
            }
        self.events[name]['start'].record()
        
    def end(self, name="default", delete=False):
        """End timing an event with the given name and return elapsed time in seconds. Optionally delete the event to free memory."""
        if name not in self.events:
            raise KeyError(f"No timing event found with name '{name}'")
        self.events[name]['end'].record()
        self.events[name]['end'].synchronize()
        elapsed = self.events[name]['start'].elapsed_time(self.events[name]['end']) / 1000.0
        if delete:
            del self.events[name]
        return elapsed
        
    def get_all_times(self):
        """Return a dictionary of all event timings"""
        times = {}
        for name in self.events:
            self.events[name]['end'].synchronize()
            times[name] = self.events[name]['start'].elapsed_time(self.events[name]['end']) / 1000.0
        return times
        
    def clear(self):
        """Clear all stored events"""
        self.events.clear()

# --- Fine-grained profiling via monkey-patching ---
class ProfiledEngine(Engine):
    def __init__(self):
        super().__init__()
        self.timer = CUDATimer()

    def run(self, requests, num_decode_req=0):
        self.timer.clear()
        self.timer.start("total")
        with torch.inference_mode():
            # --- Input tensor construction ---
            self.timer.start("input_tensor")
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
            self.timer.end("input_tensor")

            # --- KV cache allocation ---
            self.timer.start("kv_cache_alloc")
            for idx, r in enumerate(requests):
                if r.request_id not in self.kv_cache_map and idx >= num_decode_req:
                    self.kv_cache_map[r.request_id] = DistKVCache(self.pool)
            seq_lens_before = [self.kv_cache_map[r.request_id].seqlen for r in requests]
            seq_lens_before_t = torch.tensor(seq_lens_before, dtype=torch.int32, device="cuda")
            self.timer.end("kv_cache_alloc")

            self.timer.start("kv_cache_allocate_tokens")
            for idx, r in enumerate(requests):
                if idx >= num_decode_req:
                    self.kv_cache_map[r.request_id].allocate_tokens(r.prompt_length)
                elif self.kv_cache_map[r.request_id].seqlen < r.current_length:
                    self.kv_cache_map[r.request_id].allocate_tokens(1)
            seq_lens_after = [self.kv_cache_map[r.request_id].seqlen for r in requests]
            seq_lens_after_t = torch.tensor(seq_lens_after, dtype=torch.int32, device="cuda")
            self.timer.end("kv_cache_allocate_tokens")

            # --- Metadata building ---
            self.timer.start("kv_metadata")
            kv_indptr, kv_indices, kv_last_page_len = build_kv_metadata(
                [self.kv_cache_map[r.request_id] for r in requests]
            )
            self.timer.end("kv_metadata")

            # --- Planning (prefill/decoder) ---
            self.timer.start("planning")
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
            self.timer.end("planning")

            # --- Embedding ---
            self.timer.start("embedding")
            hidden = self.weights["embedding"][input_tensor]
            self.timer.end("embedding")

            # --- Per-layer forward ---
            # Instead of per-layer keys, sum across layers for each event type
            event_types = [
                "ln_attn", "qkv_proj", "rope", "kv_append", "attn_kernel",
                "o_proj", "ln_ffn", "ffn", "total"
            ]
            event_sums = {k: 0.0 for k in event_types}
            for layer in range(self.layers):
                self.timer.start(f"layer_{layer}")

                # --- LayerNorm before attention ---
                self.timer.start(f"layer_{layer}_ln_attn")
                rms = torch.sqrt(hidden.square().mean(-1, keepdim=True) + 1e-5)
                ln_attn_in = (hidden / rms).to(torch.float16) * self.weights["layernormAttn_weight"][layer]
                event_sums["ln_attn"] += self.timer.end(f"layer_{layer}_ln_attn", delete=True)
                
                # --- QKV projections ---
                self.timer.start(f"layer_{layer}_qkv_proj")
                k = ln_attn_in.matmul(self.weights["self_attn_k_proj_weight"][layer].T).view(-1, self.num_kv_heads, self.head_dim)
                v = ln_attn_in.matmul(self.weights["self_attn_v_proj_weight"][layer].T).view(-1, self.num_kv_heads, self.head_dim)
                q = ln_attn_in.matmul(self.weights["self_attn_q_proj_weight"][layer].T).view(-1, self.num_qo_heads, self.head_dim)
                event_sums["qkv_proj"] += self.timer.end(f"layer_{layer}_qkv_proj", delete=True)
                
                # --- RoPE ---
                self.timer.start(f"layer_{layer}_rope")
                flashinfer.rope.apply_rope_inplace(q, k, indptr_tensor, seq_lens_before_t, rope_theta=500_000.0)
                event_sums["rope"] += self.timer.end(f"layer_{layer}_rope", delete=True)
                
                # --- KV append ---
                self.timer.start(f"layer_{layer}_kv_append")
                batch_indices, positions = flashinfer.get_batch_indices_positions(
                    indptr_tensor, seq_lens_after_t, k.shape[0]
                )
                flashinfer.append_paged_kv_cache(
                    k, v, batch_indices, positions, self.pool.get_paged_kv_cache(layer),
                    kv_indices, kv_indptr, kv_last_page_len, kv_layout="HND"
                )
                event_sums["kv_append"] += self.timer.end(f"layer_{layer}_kv_append", delete=True)
                
                # --- Attention kernel ---
                attn_out_prefill = attn_out_decode = None
                self.timer.start(f"layer_{layer}_attn_kernel")
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
                event_sums["attn_kernel"] += self.timer.end(f"layer_{layer}_attn_kernel", delete=True)
                
                # --- O projection ---
                self.timer.start(f"layer_{layer}_o_proj")
                hidden = attn_out.matmul(self.weights["o_proj_weight"][layer].T) + hidden
                event_sums["o_proj"] += self.timer.end(f"layer_{layer}_o_proj", delete=True)
                
                # --- LayerNorm before FFN ---
                self.timer.start(f"layer_{layer}_ln_ffn")
                rms = torch.sqrt(hidden.square().mean(-1, keepdim=True) + 1e-5)
                ln_ffn_in = (hidden / rms).to(torch.float16) * self.weights["layernormFFN_weight"][layer]
                event_sums["ln_ffn"] += self.timer.end(f"layer_{layer}_ln_ffn", delete=True)
                
                # --- FFN projections and activation ---
                self.timer.start(f"layer_{layer}_ffn")
                up = ln_ffn_in.matmul(self.weights["up_proj_weight"][layer].T)
                gate = ln_ffn_in.matmul(self.weights["gate_proj_weight"][layer].T)
                ffn_out = (up * torch.nn.functional.silu(gate)).matmul(self.weights["down_proj_weight"][layer].T)
                hidden = ffn_out + hidden
                event_sums["ffn"] += self.timer.end(f"layer_{layer}_ffn", delete=True)

                event_sums["total"] += self.timer.end(f"layer_{layer}", delete=True)

            # --- Logits ---
            self.timer.start("logits")
            rms = torch.sqrt(hidden.square().mean(-1, keepdim=True) + 1e-5)
            logits = (hidden / rms).to(torch.float16) * self.weights["model_layernorm_weight"]
            logits = logits.matmul(self.weights["lm_head_weight"].T)
            sample_ids = torch.argmax(logits, dim=-1)
            last_token_indices = (indptr_tensor[1:] - 1).long()
            self.timer.end("logits")

        self.timer.end("total")
        # Compose timings dict: sum across layers for each event type
        timings = self.timer.get_all_times()
        # Overwrite per-layer events with summed events
        for k in event_types:
            timings[k] = event_sums[k]
        return sample_ids[last_token_indices].cpu(), timings