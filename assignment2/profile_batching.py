#!/usr/bin/env python3
"""
Compare generation speed of:
  - assignment2.uniform_prefill.Engine  - fixed prompt length
  - assignment2.different_prefill.Engine - varied prompt lengths within each batch

We generate 128 new tokens, sweep batch size from 2**0–2**6,
measure latency and throughput.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import torch

from uniform_prefill import Engine as UniformEngine
from different_prefill import Engine as DifferentEngine


# --------------------------------------------------------------------------- #
def build_prompt(tokenizer, target_len: int) -> str:
    """Make exactly `target_len` tokens by over-generating and slicing."""
    with open("input.txt", "r") as f:
        base = f.read()
    ids = tokenizer.encode(base)[:target_len]
    return tokenizer.decode(ids)


@torch.inference_mode()
def time_generation(
    engine,
    prompts: List[str],
    rounds: int,
) -> float:
    """Time engine.generate_batched(prompts, rounds=rounds)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    engine.generate_batched(prompts, rounds=rounds)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    ROUNDS = 128
    batch_sizes = [2**i for i in range(7)]  # [1,2,4,8,16,32,64]

    # UniformEngine (@ fixed 1024‐token prompt)
    tmp = UniformEngine()
    fixed_prompt = build_prompt(tmp.tokenizer, 1024)
    del tmp

    uni_times = []
    engine = UniformEngine()
    for B in batch_sizes:
        # warm‐up once at B=1
        if B == batch_sizes[0]:
            engine.generate_batched([fixed_prompt], rounds=1)
            torch.cuda.empty_cache()

        dt = time_generation(engine, [fixed_prompt] * B, rounds=ROUNDS)
        uni_times.append(dt)
        print(f"[Uniform] batch={B:2d}  {dt:.3f}s, throughput={B*ROUNDS/dt:.1f} tok/s")
        torch.cuda.empty_cache()
    uni_tp = [B * ROUNDS / t for B, t in zip(batch_sizes, uni_times)]
    del engine
    # ─── DifferentEngine (@ varied prompt lengths in each batch) ─────────────
    min_len, max_len = 256, 1024
    diff_times, diff_tp = [], []

    # one engine instance for all B to preserve any internal cache/kernels
    engine = DifferentEngine()
    tokenizer = engine.tokenizer
    # warm‐up on a single max‐length prompt
    warmup = build_prompt(tokenizer, max_len)
    engine.generate_batched([warmup], rounds=1)

    for B in batch_sizes:
        # linearly spaced prompt lengths between min_len and max_len
        lengths = np.linspace(min_len, max_len, B, dtype=int)
        
        prompts = [build_prompt(tokenizer, L) for L in lengths]

        dt = time_generation(engine, prompts, rounds=ROUNDS)
        diff_times.append(dt)
        diff_tp.append(B * ROUNDS / dt)
        print(
            f"[Different] batch={B:2d} {dt:.3f}s, throughput={B*ROUNDS/dt:.1f} tok/s"
        )

    del engine
    torch.cuda.empty_cache()

    # Plot latency
    plt.figure(figsize=(8, 5))
    plt.plot(batch_sizes, uni_times, "o-", label="Uniform (1024)")
    plt.plot(batch_sizes, diff_times, "o-", label="Different (256→1024)")
    plt.xscale("log", base=2)
    plt.xlabel("Batch size")
    plt.ylabel("Latency (s)")
    plt.title("Generation Latency @128 new tokens")
    plt.legend()
    plt.tight_layout()
    plt.savefig("latency.png", dpi=150)

    # Plot throughput
    plt.figure(figsize=(8, 5))
    plt.plot(batch_sizes, uni_tp, "o-", label="Uniform (1024)")
    plt.plot(batch_sizes, diff_tp, "o-", label="Different (256→1024)")
    plt.xscale("log", base=2)
    plt.xlabel("Batch size")
    plt.ylabel("Throughput (tokens/s)")
    plt.title("Generation Throughput @128 new tokens")
    plt.legend()
    plt.tight_layout()
    plt.savefig("throughput.png", dpi=150)

    print("\nSaved plots latency.png, throughput.png")


if __name__ == "__main__":
    main()
