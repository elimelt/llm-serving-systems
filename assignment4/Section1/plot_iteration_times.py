import numpy as np
import matplotlib.pyplot as plt

# Load per-iteration times
chunked_times = np.load('results/chunked_times.npy')
continous_times = np.load('results/continous_times.npy')
naive_times = np.load("results/naive_times.npy")


def plot_iteration_times(times, label):
    plt.scatter(range(len(times)), times, s=5, alpha=0.2, label=label)
    plt.xlabel('Iteration ID')
    plt.ylabel('Iteration Time (ms)')
    plt.yscale('log')
    plt.title('Iteration Time Comparison: Chunked Prefill vs. Continuous Batching')

plt.figure(figsize=(12, 6))
plot_iteration_times(naive_times, "Naive Scheduler")
plot_iteration_times(continous_times, "Continuous Scheduler")
plt.legend()
plt.tight_layout()
plt.savefig('plots/iteration_times_comparison_naive_continous.png')
plt.show()

plt.figure(figsize=(12, 6))
plot_iteration_times(chunked_times, "Chunked Prefill")
plot_iteration_times(continous_times, "Continuous Batching")
plt.legend()
plt.tight_layout()
plt.savefig('plots/iteration_times_comparison_chunked_continous.png')
plt.show()