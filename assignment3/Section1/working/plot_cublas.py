import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("cublas_perf.csv")
os.makedirs("plots", exist_ok=True)


def plot_matrix():
    unique_N = sorted(df["N"].unique())
    unique_K = sorted(df["K"].unique())

    for n in unique_N:
        for k in unique_K:
            subset = df[(df["N"] == n) & (df["K"] == k)]

            if subset.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(subset["M"], subset["tflops"], "o-", linewidth=2, markersize=8)

            ax.set_xlabel("M", fontsize=12)
            ax.set_ylabel("TFLOPS", fontsize=12)
            ax.set_title(f"cuBLAS GEMM (N={n}, K={k})", fontsize=14)

            ax.grid(True, linestyle="--", alpha=0.7)

            # Add data points as text labels
            # for i, row in subset.iterrows():
            #     ax.annotate(
            #         f"{row['tflops']:.2f}",
            #         (row["M"], row["tflops"]),
            #         textcoords="offset points",
            #         xytext=(0, 10),
            #         ha="center",
            #     )

            filename = f"plots/GEMM_N{n}_K{k}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close(fig)

    # Create a summary plot showing all data
    plt.figure(figsize=(12, 8))

    # Use different marker/color for each N,K combination
    for n in unique_N:
        for k in unique_K:
            subset = df[(df["N"] == n) & (df["K"] == k)]
            if not subset.empty:
                plt.plot(
                    subset["M"],
                    subset["tflops"],
                    "o-",
                    label=f"N={n}, K={k}",
                    linewidth=2,
                    markersize=6,
                )

    plt.xlabel("Matrix Dimension (M)", fontsize=12)
    plt.ylabel("TFLOPS", fontsize=12)
    plt.title("cuBLAS GEMM TFLOPS", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=10)
    plt.savefig("plots/GEMM_summary.png", dpi=300, bbox_inches="tight")
    plt.close()


# Execute the plotting function
plot_matrix()
print("Plots generated in 'plots' directory")
