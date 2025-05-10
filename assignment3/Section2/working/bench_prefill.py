import math
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from bench_helper import get_model_configs, DTYPE, DEVICE, BYTES_PER_ELEM

# Import FlashAttention-3 and FlashInfer APIs
from flash_attn_interface import flash_attn_qkvpacked_func, flash_attn_func
import flashinfer

MODELS = get_model_configs()
ITERS = 100
WARMUP = 10

def compute_tflops(H_qo, H_kv, d, p, elapsed_s, batch_size=1, b=BYTES_PER_ELEM):
    # Compute utilization (TFLOPs) as per user formula, accounting for batch size
    flops = batch_size * (H_qo * 4 * d * p ** 2 + H_qo * 3 * p ** 2)
    bytes = batch_size * (b * p * (2 * d * H_qo + 2 * d * H_kv))
    tflops = flops / elapsed_s / 1e12
    op_intensity = flops / bytes if bytes > 0 else 0
    return tflops, op_intensity


def benchmark_flashattn3_qkvpacked(model_cfg, batch_size, seq_len, n_repeats=ITERS):
    """Benchmark FlashAttention-3 prefill attention (QKV packed)."""
    H_qo = model_cfg['num_qo_heads']
    d = model_cfg['head_dim']
    qkv = torch.randn(batch_size, seq_len, 3, H_qo, d, device=DEVICE, dtype=DTYPE)
    torch.cuda.synchronize()
    for _ in range(WARMUP):
        flash_attn_qkvpacked_func(qkv, 0.0, causal=True)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_repeats):
        flash_attn_qkvpacked_func(qkv, 0.0, causal=True)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / n_repeats
    return elapsed


def benchmark_flashattn3_unpacked(model_cfg, batch_size, seq_len, n_repeats=ITERS):
    """Benchmark FlashAttention-3 prefill attention (Q, K, V unpacked)."""
    H_qo = model_cfg['num_qo_heads']
    d = model_cfg['head_dim']
    q = torch.randn(batch_size, seq_len, H_qo, d, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch_size, seq_len, H_qo, d, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch_size, seq_len, H_qo, d, device=DEVICE, dtype=DTYPE)
    torch.cuda.synchronize()
    for _ in range(WARMUP):
        flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_repeats):
        flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / n_repeats
    return elapsed


def benchmark_flashinfer(model_cfg, batch_size, seq_len, n_repeats=ITERS):
    """Benchmark FlashInfer prefill attention (using BatchPrefillWithRaggedKVCacheWrapper, correct for batch sweep)."""
    H_qo = model_cfg['num_qo_heads']
    H_kv = model_cfg['num_kv_heads']
    d = model_cfg['head_dim']
    device = DEVICE
    dtype = DTYPE

    # For ragged, each batch has its own contiguous block of length seq_len
    nnz_qo = batch_size * seq_len
    nnz_kv = batch_size * seq_len
    qo_indptr = torch.arange(0, nnz_qo + 1, seq_len, dtype=torch.int32, device=device)
    kv_indptr = qo_indptr.clone()

    # Q, K, V shapes: [nnz_qo, H_qo, d], [nnz_kv, H_kv, d], [nnz_kv, H_kv, d]
    q = torch.randn(nnz_qo, H_qo, d, device=device, dtype=dtype)
    k = torch.randn(nnz_kv, H_kv, d, device=device, dtype=dtype)
    v = torch.randn(nnz_kv, H_kv, d, device=device, dtype=dtype)

    # Allocate workspace buffer (128MB)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace_buffer, "NHD")

    # Plan (setup auxiliary data structures)
    prefill_wrapper.plan(
        qo_indptr,
        kv_indptr,
        H_qo,
        H_kv,
        d,
        causal=True,
    )

    # Warmup
    torch.cuda.synchronize()
    for _ in range(WARMUP):
        prefill_wrapper.run(q, k, v)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(n_repeats):
        prefill_wrapper.run(q, k, v)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / n_repeats
    return elapsed


def run_benchmarks(fa_mode='qkvpacked'):
    """
    fa_mode: 'qkvpacked' or 'unpacked'.
    """
    results = {}
    for model_name, cfg in MODELS.items():
        results[model_name] = {'p_sweep': [], 'batch_sweep': []}
        # Sequence length sweep (batch=1)
        if model_name == 'LLaMA2-7B':
            p_range = [2 ** i for i in range(7, 13)]
        else:
            p_range = [2 ** i for i in range(7, 16)]
        tflops_fa_p, tflops_fi_p = [], []
        for p in p_range:
            try:
                if fa_mode == 'qkvpacked':
                    t_fa = benchmark_flashattn3_qkvpacked(cfg, 1, p)
                else:
                    t_fa = benchmark_flashattn3_unpacked(cfg, 1, p)
                t_fi = benchmark_flashinfer(cfg, 1, p)
                tf_fa, _ = compute_tflops(cfg['num_qo_heads'], cfg['num_kv_heads'], cfg['head_dim'], p, t_fa, batch_size=1)
                tf_fi, _ = compute_tflops(cfg['num_qo_heads'], cfg['num_kv_heads'], cfg['head_dim'], p, t_fi, batch_size=1)
            except RuntimeError:
                tf_fa, tf_fi = float('nan'), float('nan')
            tflops_fa_p.append(tf_fa)
            tflops_fi_p.append(tf_fi)
        results[model_name]['p_sweep'] = {
            'p': p_range,
            'FlashAttention-3': tflops_fa_p,
            'FlashInfer': tflops_fi_p,
        }
        # Batch size sweep (p=1024)
        p = 1024
        batch_range = [2 ** i for i in range(0, 7)]
        tflops_fa_batch, tflops_fi_batch = [], []
        for b in batch_range:
            try:
                if fa_mode == 'qkvpacked':
                    t_fa = benchmark_flashattn3_qkvpacked(cfg, b, p)
                else:
                    t_fa = benchmark_flashattn3_unpacked(cfg, b, p)
                t_fi = benchmark_flashinfer(cfg, b, p)
                tf_fa, _ = compute_tflops(cfg['num_qo_heads'], cfg['num_kv_heads'], cfg['head_dim'], p, t_fa, batch_size=b)
                tf_fi, _ = compute_tflops(cfg['num_qo_heads'], cfg['num_kv_heads'], cfg['head_dim'], p, t_fi, batch_size=b)
            except RuntimeError:
                tf_fa, tf_fi = float('nan'), float('nan')
            tflops_fa_batch.append(tf_fa)
            tflops_fi_batch.append(tf_fi)
        results[model_name]['batch_sweep'] = {
            'batch': batch_range,
            'FlashAttention-3': tflops_fa_batch,
            'FlashInfer': tflops_fi_batch,
        }
    return results


def plot_results(results):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharey='row')
    model_names = list(MODELS.keys())
    # Sequence length sweep
    for i, model in enumerate(model_names):
        ax = axs[0, i]
        p = results[model]['p_sweep']['p']
        fa = results[model]['p_sweep']['FlashAttention-3']
        fi = results[model]['p_sweep']['FlashInfer']
        ax.plot(np.log2(p), fa, label='FlashAttention-3', marker='o')
        ax.plot(np.log2(p), fi, label='FlashInfer', marker='x')
        ax.set_xlabel('log2(p) (sequence length)')
        ax.set_title(model)
        ax.set_xticks(np.log2(p))
        ax.set_xticklabels([str(x) for x in p])
        ax.grid(True, which='both')
        if i == 0:
            ax.set_ylabel('Compute Utilization (TFLOPs)')
        ax.legend()
    # Batch size sweep
    for i, model in enumerate(model_names):
        ax = axs[1, i]
        b = results[model]['batch_sweep']['batch']
        fa = results[model]['batch_sweep']['FlashAttention-3']
        fi = results[model]['batch_sweep']['FlashInfer']
        ax.plot(np.log2(b), fa, label='FlashAttention-3', marker='o')
        ax.plot(np.log2(b), fi, label='FlashInfer', marker='x')
        ax.set_xlabel('log2(batch size)')
        ax.set_title(model)
        ax.set_xticks(np.log2(b))
        ax.set_xticklabels([str(x) for x in b])
        ax.grid(True, which='both')
        if i == 0:
            ax.set_ylabel('Compute Utilization (TFLOPs)')
        ax.legend()
    fig.suptitle('Prefill Attention Compute Utilization per Layer')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('prefill_attention_compute_utilization.png')


def main():
    # Choose 'qkvpacked' or 'unpacked' for FlashAttention-3 benchmarking
    fa_mode = 'qkvpacked'  # or 'unpacked'
    results = run_benchmarks(fa_mode=fa_mode)
    plot_results(results)


if __name__ == '__main__':
    main()
