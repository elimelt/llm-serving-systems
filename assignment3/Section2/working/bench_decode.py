import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from bench_helper import get_model_configs, DTYPE, DEVICE, BYTES_PER_ELEM

from flash_attn_interface import flash_attn_with_kvcache
import flashinfer

MODELS = get_model_configs()

WARMUP = 10
ITERS = 100

def compute_decode_bandwidth(H_qo, H_kv, d, c, elapsed_s, batch_size=1, b=BYTES_PER_ELEM):
    bytes = batch_size * (b * d * (2 * H_qo + 2 * c * H_kv))
    gbps = bytes / elapsed_s / 1e9
    return gbps


def benchmark_flashattn3_decode(model_cfg, batch_size, context_len, page_size=64, n_repeats=ITERS):
    H_qo = model_cfg['num_qo_heads']
    H_kv = model_cfg['num_kv_heads']
    d = model_cfg['head_dim']
    q = torch.randn(batch_size, 1, H_qo, d, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch_size, context_len, H_kv, d, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch_size, context_len, H_kv, d, device=DEVICE, dtype=DTYPE)
    torch.cuda.synchronize()
    for _ in range(WARMUP):
        flash_attn_with_kvcache(q, k, v, causal=True)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(n_repeats):
        flash_attn_with_kvcache(q, k, v, causal=True)
    end_event.record()
    end_event.synchronize()
    elapsed = start_event.elapsed_time(end_event) / 1000.0 / n_repeats
    return elapsed


def benchmark_flashinfer_decode(model_cfg, batch_size, context_len, page_size=64, n_repeats=ITERS):
    H_qo = model_cfg['num_qo_heads']
    H_kv = model_cfg['num_kv_heads']
    d = model_cfg['head_dim']
    device = DEVICE
    dtype = DTYPE

    num_pages_per_seq = math.ceil(context_len / page_size)
    max_num_pages = batch_size * num_pages_per_seq

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD", use_tensor_cores=True)

    kv_page_indptr = torch.arange(0, (batch_size + 1) * num_pages_per_seq, num_pages_per_seq, dtype=torch.int32, device=device)
    kv_page_indices = torch.arange(max_num_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full((batch_size,), page_size, dtype=torch.int32, device=device)
    for i in range(batch_size):
        rem = context_len % page_size
        if rem != 0:
            kv_last_page_len[i] = rem

    kv_cache = torch.randn(max_num_pages, 2, page_size, H_kv, d, dtype=dtype, device=device)

    decode_wrapper.plan(
        kv_page_indptr,
        kv_page_indices,
        kv_last_page_len,
        H_qo,
        H_kv,
        d,
        page_size,
        pos_encoding_mode="NONE",
        data_type=dtype
    )

    torch.cuda.synchronize()
    for _ in range(WARMUP):
        q = torch.randn(batch_size, H_qo, d, dtype=dtype, device=device)
        decode_wrapper.run(q, kv_cache)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(n_repeats):
        q = torch.randn(batch_size, H_qo, d, dtype=dtype, device=device)
        decode_wrapper.run(q, kv_cache)
    end_event.record()
    end_event.synchronize()
    elapsed = start_event.elapsed_time(end_event) / 1000.0 / n_repeats
    return elapsed


def run_decode_benchmarks():
    results = {}
    for model_name, cfg in MODELS.items():
        results[model_name] = {'context_sweep': [], 'batch_sweep': [], 'page_sweep': []}
        # Sweep 1: Context length (batch=1, page_size=16)
        if model_name == 'LLaMA2-7B':
            c_range = [2 ** i for i in range(7, 13)]
        else:
            c_range = [2 ** i for i in range(7, 16)]
        gbps_fa, gbps_fi = [], []
        for c in c_range:
            try:
                t_fa = benchmark_flashattn3_decode(cfg, 1, c, page_size=16)
                t_fi = benchmark_flashinfer_decode(cfg, 1, c, page_size=16)
                bw_fa = compute_decode_bandwidth(cfg['num_qo_heads'], cfg['num_kv_heads'], cfg['head_dim'], c, t_fa, batch_size=1)
                bw_fi = compute_decode_bandwidth(cfg['num_qo_heads'], cfg['num_kv_heads'], cfg['head_dim'], c, t_fi, batch_size=1)
            except RuntimeError:
                bw_fa, bw_fi = float('nan'), float('nan')
            gbps_fa.append(bw_fa)
            gbps_fi.append(bw_fi)
        results[model_name]['context_sweep'] = {
            'c': c_range,
            'FlashAttention-3': gbps_fa,
            'FlashInfer': gbps_fi,
        }
        # Sweep 2: Batch size (c=1024, page_size=16)
        c = 1024
        batch_range = [2 ** i for i in range(0, 7)]
        gbps_fa, gbps_fi = [], []
        for b in batch_range:
            try:
                t_fa = benchmark_flashattn3_decode(cfg, b, c, page_size=16)
                t_fi = benchmark_flashinfer_decode(cfg, b, c, page_size=16)
                bw_fa = compute_decode_bandwidth(cfg['num_qo_heads'], cfg['num_kv_heads'], cfg['head_dim'], c, t_fa, batch_size=b)
                bw_fi = compute_decode_bandwidth(cfg['num_qo_heads'], cfg['num_kv_heads'], cfg['head_dim'], c, t_fi, batch_size=b)
            except RuntimeError:
                bw_fa, bw_fi = float('nan'), float('nan')
            gbps_fa.append(bw_fa)
            gbps_fi.append(bw_fi)
        results[model_name]['batch_sweep'] = {
            'batch': batch_range,
            'FlashAttention-3': gbps_fa,
            'FlashInfer': gbps_fi,
        }
        # Sweep 3: Page size (batch=128, c=1024, page_size=1,2,4,8,16)
        b = 128
        c = 1024
        page_sizes = [1, 2, 4, 8, 16]
        gbps_fa, gbps_fi = [], []
        for psize in page_sizes:
            try:
                t_fa = benchmark_flashattn3_decode(cfg, b, c, page_size=psize)
                t_fi = benchmark_flashinfer_decode(cfg, b, c, page_size=psize)
                bw_fa = compute_decode_bandwidth(cfg['num_qo_heads'], cfg['num_kv_heads'], cfg['head_dim'], c, t_fa, batch_size=b)
                bw_fi = compute_decode_bandwidth(cfg['num_qo_heads'], cfg['num_kv_heads'], cfg['head_dim'], c, t_fi, batch_size=b)
            except RuntimeError:
                bw_fa, bw_fi = float('nan'), float('nan')
            gbps_fa.append(bw_fa)
            gbps_fi.append(bw_fi)
        results[model_name]['page_sweep'] = {
            'page_size': page_sizes,
            'FlashAttention-3': gbps_fa,
            'FlashInfer': gbps_fi,
        }
    return results


def plot_decode_results(results):
    fig, axs = plt.subplots(3, 3, figsize=(18, 15), sharey='row')
    model_names = list(MODELS.keys())
    # Context length sweep
    for i, model in enumerate(model_names):
        ax = axs[0, i]
        c = results[model]['context_sweep']['c']
        fa = results[model]['context_sweep']['FlashAttention-3']
        fi = results[model]['context_sweep']['FlashInfer']
        ax.plot(np.log2(c), fa, label='FlashAttention-3', marker='o')
        ax.plot(np.log2(c), fi, label='FlashInfer', marker='x')
        ax.set_xlabel('context length')
        ax.set_title(model)
        ax.set_xticks(np.log2(c))
        ax.set_xticklabels([str(x) for x in c])
        ax.grid(True, which='both')
        if i == 0:
            ax.set_ylabel('Memory Bandwidth Utilization (GB/s)')
        ax.legend()
    # Batch size sweep
    for i, model in enumerate(model_names):
        ax = axs[1, i]
        b = results[model]['batch_sweep']['batch']
        fa = results[model]['batch_sweep']['FlashAttention-3']
        fi = results[model]['batch_sweep']['FlashInfer']
        ax.plot(np.log2(b), fa, label='FlashAttention-3', marker='o')
        ax.plot(np.log2(b), fi, label='FlashInfer', marker='x')
        ax.set_xlabel('batch size')
        ax.set_title(model)
        ax.set_xticks(np.log2(b))
        ax.set_xticklabels([str(x) for x in b])
        ax.grid(True, which='both')
        if i == 0:
            ax.set_ylabel('Memory Bandwidth Utilization (GB/s)')
        ax.legend()
    # Page size sweep
    for i, model in enumerate(model_names):
        ax = axs[2, i]
        psize = results[model]['page_sweep']['page_size']
        fa = results[model]['page_sweep']['FlashAttention-3']
        fi = results[model]['page_sweep']['FlashInfer']
        ax.plot(np.log2(psize), fa, label='FlashAttention-3', marker='o')
        ax.plot(np.log2(psize), fi, label='FlashInfer', marker='x')
        ax.set_xlabel('Page size')
        ax.set_title(model)
        ax.set_xticks(np.log2(psize))
        ax.set_xticklabels([str(x) for x in psize])
        ax.grid(True, which='both')
        if i == 0:
            ax.set_ylabel('Memory Bandwidth Utilization (GB/s)')
        ax.legend()
    fig.suptitle('Decode Attention Memory Bandwidth Utilization per Layer')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('decode_attention_bandwidth.png')


def main():
    results = run_decode_benchmarks()
    plot_decode_results(results)

if __name__ == '__main__':
    main() 