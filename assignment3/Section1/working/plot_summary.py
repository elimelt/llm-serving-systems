import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d

# N,M,K,library,tflops
df = pd.read_csv("cublas_perf.csv")
os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(12, 8))

# Get all unique (N, K) shapes from both cublas and cutlass
shapes = set(tuple(x) for x in df[['N', 'K']].drop_duplicates().values.tolist())

# Also add shapes from cutlass files
results_dir = "results"
for fname in os.listdir(results_dir):
    if fname.endswith(".csv") and fname.startswith("gemm_"):
        parts = fname.split("_")
        if len(parts) >= 3:
            try:
                file_N = int(parts[1])
                file_K = int(parts[2].split(".")[0])
                shapes.add((file_N, file_K))
            except Exception:
                continue

# Assign a color to each shape
import itertools
colors = plt.cm.tab20.colors
color_cycle = itertools.cycle(colors)
shape_to_color = {}
for shape in sorted(shapes):
    shape_to_color[shape] = next(color_cycle)

# --- Plot CUBLAS ---
for (N, K) in sorted(shapes):
    shape_df = df[(df['N'] == N) & (df['K'] == K) & (df['library'] == 'cublas')]
    if not shape_df.empty:
        x = shape_df['M'] * shape_df['N'] * shape_df['K']
        y = shape_df['tflops']
        # Sort by x for line plotting
        sort_idx = np.argsort(x)
        x = x.iloc[sort_idx]
        y = y.iloc[sort_idx]
        plt.plot(x, y, '-', color=shape_to_color[(N, K)], label=f'CUBLAS N={N} K={K}')

# --- Plot CUTLASS ---
for (N, K) in sorted(shapes):
    # Find the corresponding file
    cutlass_file = None
    for fname in os.listdir(results_dir):
        if fname.endswith(".csv") and fname.startswith("gemm_"):
            parts = fname.split("_")
            if len(parts) >= 3:
                try:
                    file_N = int(parts[1])
                    file_K = int(parts[2].split(".")[0])
                    if file_N == N and file_K == K:
                        cutlass_file = os.path.join(results_dir, fname)
                        break
                except Exception:
                    continue
    if cutlass_file is not None:
        df_cutlass = pd.read_csv(cutlass_file)
        # For each M, take the best kernel (max GFLOPs)
        idx = df_cutlass.groupby('m')['GFLOPs'].idxmax()
        best_kernels = df_cutlass.loc[idx]
        x = best_kernels['m'] * best_kernels['n'] * best_kernels['k']
        y = best_kernels['GFLOPs'] / 1000
        # Sort by x for line plotting
        sort_idx = np.argsort(x)
        x = x.iloc[sort_idx]
        y = y.iloc[sort_idx]
        plt.plot(x, y, '--', color=shape_to_color[(N, K)], label=f'CUTLASS N={N} K={K}')

plt.title('GEMM Performance vs Total Problem Size')
plt.xlabel('Total Number of Elements (M x N x K)')
plt.ylabel('TFLOPS')
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.legend(ncol=2, fontsize='small')
plt.tight_layout()

plt.savefig("plots/GEMM_all_elements_with_average.png", dpi=300, bbox_inches="tight")
print("Plot saved to plots/GEMM_all_elements_with_average.png")
