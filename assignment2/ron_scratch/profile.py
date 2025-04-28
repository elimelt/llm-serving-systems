#!/usr/bin/env python3
"""
Compare generation speed of:
  - assignment2.no_kv.Engine  - baseline (no KV cache)
  - assignment2.single_batch.Engine - with KV cache

We fix the prompt at 1024 tokens, then vary the number of generated
tokens from 128 to 2048 (inclusive, step 128).  Timings exclude
model-loading and one-off warm-up overhead.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch

from no_kv import Engine as EngineNoKV
from single_batch import Engine as EngineKV


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def build_prompt(tokenizer, target_len: int = 1024) -> str:
    """
    Construct a prompt whose *encoded* length is exactly `target_len` tokens.
    We over-generate simple text and slice the resulting IDs to length N.
    """
    with open("../input.txt", "r") as f:
        base = f.read()
    ids = tokenizer.encode(base)[:target_len]          # trim to N
    return tokenizer.decode(ids)


@torch.inference_mode()
def time_generation(
    engine,
    prompt: str,
    num_new: int,
) -> float:
    """
    Time a single call to `engine.generate(prompt, rounds=num_new)`.

    Returns
    -------
    float  - elapsed seconds on wall-clock.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    engine.generate(prompt, rounds=num_new)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def benchmark(
    engine_cls,
    prompt: str,
    output_lengths: List[int],
    label: str,
) -> List[float]:
    """
    Instantiate `engine_cls`, warm-up once, then time every output length.
    """
    engine = engine_cls()                     # model load (not timed later)

    # warm-up (1 token) to build CUDA kernels & caches
    engine.generate(prompt, rounds=1)

    print(f"timings for {label}:")
    times: List[float] = []
    for out_len in output_lengths:
        dt = time_generation(engine, prompt, out_len)
        times.append(dt)
        print(f"  {out_len:4d} tokens: {dt:.3f} s")
    del engine
    torch.cuda.empty_cache()
    return times, label


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    # common prompt of exactly 1024 tokens
    tmp_engine = EngineNoKV()
    prompt = build_prompt(tmp_engine.tokenizer, 1024)
    del tmp_engine

    out_lens = list(range(128, 2049, 128))

    kv_times, kv_label = benchmark(EngineKV, prompt, out_lens, "with KV cache")
    no_kv_times, no_kv_label = benchmark(EngineNoKV, prompt, out_lens, "no KV cache")

    # ---------------------------  plot  ------------------------------------ #
    plt.figure(figsize=(8, 5))
    plt.plot(out_lens, no_kv_times, marker="o", label=no_kv_label)
    plt.plot(out_lens, kv_times, marker="o", label=kv_label)
    plt.xlabel("output tokens generated")
    plt.ylabel("elapsed time (s)")
    plt.title("Generation time vs. output length (prompt = 1024 tokens)")
    plt.legend()
    plt.tight_layout()

    # save to disk as well (handy when running on headless server)
    out_path = Path("kv_vs_no_kv.png")
    plt.savefig(out_path, dpi=150)
    print(f"plot saved -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
