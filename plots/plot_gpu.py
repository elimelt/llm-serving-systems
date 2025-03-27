import matplotlib.pyplot as plt
import numpy as np

# GPU data as lists
vendors = ["NVIDIA", "NVIDIA", "NVIDIA", "NVIDIA", "NVIDIA", "NVIDIA", "NVIDIA",
           "AMD", "AMD", "AMD", "Intel", "Intel"]
models = ["V100", "A100-40GB", "A100-80GB", "H100", "H200", "B100", "B200",
          "MI250", "MI300", "MI325X", "Gaudi 2", "Gaudi 3"]
years = [2017, 2020, 2021, 2023, 2024, 2024, 2024,
         2021, 2023, 2024, 2022, 2024]
compute = [125000, 312000, 312000, 989000, 989000, 1800000, 2250000,
           362000, 1307000, 1307000, 1000000, 1800000]
memory_size = [16, 40, 80, 80, 96, 120, 120, 128, 192, 256, 96, 128]
memory_bw = [900, 1555, 2000, 3352, 4800, 8000, 8000, 3352, 5300, 6000, 2400, 3700]
net_bw = [300, 600, 600, 900, 900, 1800, 1800, 800, 1024, 1024, 600, 1200]
colors = {"NVIDIA": "blue", "AMD": "red", "Intel": "green"}

# Plotting three scatter plots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Year vs Compute
axs[0].set_title("Year-Compute Scatter Plot of GPU Models")
axs[0].set_xlabel("Release Year")
axs[0].set_ylabel("Compute (FP16 GFLOP/s)")
plotted_vendors = set()
for vendor, model, year, c in zip(vendors, models, years, compute):
    if vendor not in plotted_vendors:
        axs[0].scatter(year, c, color=colors[vendor], label=vendor, s=100)
        plotted_vendors.add(vendor)
    else:
        axs[0].scatter(year, c, color=colors[vendor], s=100)
    axs[0].text(year, c, model, fontsize=12, ha='right', va='bottom')
axs[0].legend(title="Vendor")
axs[0].grid(True)

# Plot 2: Year vs Memory Size
axs[1].set_title("Year-Memory Size Scatter Plot of GPU Models")
axs[1].set_xlabel("Release Year")
axs[1].set_ylabel("Memory Size (GB)")
plotted_vendors = set()
for vendor, model, year, ms in zip(vendors, models, years, memory_size):
    if vendor not in plotted_vendors:
        axs[1].scatter(year, ms, color=colors[vendor], label=vendor, s=100)
        plotted_vendors.add(vendor)
    else:
        axs[1].scatter(year, ms, color=colors[vendor], s=100)
    axs[1].text(year, ms, model, fontsize=12, ha='right', va='bottom')
axs[1].legend(title="Vendor")
axs[1].grid(True)

# Plot 3: Year vs Memory Bandwidth
axs[2].set_title("Year-Memory Bandwidth Scatter Plot of GPU Models")
axs[2].set_xlabel("Release Year")
axs[2].set_ylabel("Memory Bandwidth (GB/s)")
plotted_vendors = set()
for vendor, model, year, bw in zip(vendors, models, years, memory_bw):
    if vendor not in plotted_vendors:
        axs[2].scatter(year, bw, color=colors[vendor], label=vendor, s=100)
        plotted_vendors.add(vendor)
    else:
        axs[2].scatter(year, bw, color=colors[vendor], s=100)
    axs[2].text(year, bw, model, fontsize=12, ha='right', va='bottom')
axs[2].legend(title="Vendor")
axs[2].grid(True)

# Adjust layout and show the plots
plt.tight_layout()
plt.savefig("gpu_scatter_plots.png", dpi=300)
