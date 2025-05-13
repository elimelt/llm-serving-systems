import pandas as pd
import matplotlib.pyplot as plt
import os

# N,M,K,average_time,tflops
df = pd.read_csv("cublas_perf.csv")
os.makedirs("plots", exist_ok=True)


# Get all unique (N, K) shapes
shapes = df[['N', 'K']].drop_duplicates().values.tolist()

results_dir = "results"

for N, K in shapes:
    shape_df = df[(df['N'] == N) & (df['K'] == K)]
    batch_sizes = sorted(shape_df['M'].unique())

    plt.figure(figsize=(8, 5))

    # Plot cublas results
    cublas_df = shape_df[shape_df['library'] == 'cublas']
    if not cublas_df.empty:
        plt.plot(cublas_df['M'], cublas_df['tflops'], marker='o', label='CUBLAS')

    # Plot cutlass results from results/ directory
    # Find the corresponding file
    cutlass_file = None
    for fname in os.listdir(results_dir):
        if fname.endswith(".csv") and fname.startswith("gemm_"):
            # Parse N and K from filename
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
        # Group by M, take the max GFLOPs for each M (best kernel)
        cutlass_grouped = df_cutlass.groupby('m')['GFLOPs'].max().reset_index()
        plt.plot(cutlass_grouped['m'], cutlass_grouped['GFLOPs'] / 1000, marker='o', label='Cutlass')

    plt.title(f'Performance for Shape N={N}, K={K}')
    plt.xlabel('Matrix Dimension (M)')
    plt.ylabel('TFLOPS')
    # plt.ylim(0.0001, 60)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/GEMM_N{N}_K{K}.png", dpi=300, bbox_inches="tight")
