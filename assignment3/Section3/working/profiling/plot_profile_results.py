import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure plots output directory exists
output_dir = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(output_dir, exist_ok=True)

# 1. exp1.csv: decode_len, prefill_time, decode_time, total_time, ...
exp1 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'profile_results/exp1.csv'))

# Plot 1a: End-to-end time vs log2(decode_len)
plt.figure(figsize=(8, 5))
plt.plot(np.log2(exp1['decode_len']), exp1['prefill_time'], 'o-', label='Prefill Time')
plt.plot(np.log2(exp1['decode_len']), exp1['decode_time'], 'o-', label='Total Decode Time')
plt.plot(np.log2(exp1['decode_len']), exp1['total_time'], 'o-', label='End-to-End Time')
plt.xlabel('log2(Decode Length)')
plt.ylabel('Time (s)')
plt.title('End-to-End Time vs Decode Length (Batch=128, Prefill=1024)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'exp1_end_to_end_time.png'))
plt.close()

# Plot 1b: Area chart breakdown of decode phase (per-operation, sum across all layers)
decode_ops = [c for c in exp1.columns if c.startswith('decode_') and not any(
    c.endswith(suffix) for suffix in ['_time', '_total', '_fine', '_len', '_embedding', '_logits', '_layers']) and c != 'decode_len']
if decode_ops:
    # Calculate total contribution for each op
    op_totals = {op: exp1[op].sum() for op in decode_ops}
    top5_ops = sorted(op_totals, key=op_totals.get, reverse=True)[:5]
    other_ops = [op for op in decode_ops if op not in top5_ops]
    # Sort top5_ops by total (descending)
    top5_ops_sorted = sorted(top5_ops, key=lambda op: op_totals[op], reverse=True)
    stack_data = [exp1[op] for op in top5_ops_sorted]
    if other_ops:
        other_sum = exp1[other_ops].sum(axis=1)
        stack_data.append(other_sum)
        labels = [op.replace('decode_', '') for op in top5_ops_sorted] + ['Other']
    else:
        labels = [op.replace('decode_', '') for op in top5_ops_sorted]
    n = len(stack_data)
    cmap = plt.cm.get_cmap('viridis', n)
    colors = [cmap(i) for i in range(n)]
    plt.figure(figsize=(10, 6))
    stack = plt.stackplot(
        np.log2(exp1['decode_len']),
        stack_data,
        labels=labels,
        alpha=1.0,
        colors=colors
    )
    plt.xlabel('log2(Decode Length)')
    plt.ylabel('Time (s)')
    plt.title('Decode Phase Operation Breakdown (Batch=128, Prefill=1024)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp1_decode_breakdown_area.png'))
    plt.close()

# Plot 1c: Area chart breakdown of last decode step (per-operation, sum across all layers)
last_decode_ops = [c for c in exp1.columns if c.startswith('last_decode_') and not any(
    c.endswith(suffix) for suffix in ['_time', '_total', '_fine', '_len', '_embedding', '_logits', '_layers']) and c != 'last_decode_len']
if last_decode_ops:
    op_totals = {op: exp1[op].sum() for op in last_decode_ops}
    top5_ops = sorted(op_totals, key=op_totals.get, reverse=True)[:5]
    other_ops = [op for op in last_decode_ops if op not in top5_ops]
    top5_ops_sorted = sorted(top5_ops, key=lambda op: op_totals[op], reverse=True)
    stack_data = [exp1[op] for op in top5_ops_sorted]
    if other_ops:
        other_sum = exp1[other_ops].sum(axis=1)
        stack_data.append(other_sum)
        labels = [op.replace('last_decode_', '') for op in top5_ops_sorted] + ['Other']
    else:
        labels = [op.replace('last_decode_', '') for op in top5_ops_sorted]
    n = len(stack_data)
    cmap = plt.cm.get_cmap('plasma', n)
    colors = [cmap(i) for i in range(n)]
    plt.figure(figsize=(10, 6))
    stack = plt.stackplot(
        np.log2(exp1['decode_len']),
        stack_data,
        labels=labels,
        alpha=1.0,
        colors=colors
    )
    plt.xlabel('log2(Decode Length)')
    plt.ylabel('Time (s)')
    plt.title('Last Decode Step Operation Breakdown (Batch=128, Prefill=1024)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp1_last_decode_breakdown_area.png'))
    plt.close()

# 2. exp2.csv: prefill_len, prefill_time, ...
exp2 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'profile_results/exp2.csv'))

# Plot 2a: Prefill time vs log2(prefill_len)
plt.figure(figsize=(8, 5))
plt.plot(np.log2(exp2['prefill_len']), exp2['prefill_time'], 'o-', label='Prefill Time')
plt.xlabel('log2(Prefill Length)')
plt.ylabel('Time (s)')
plt.title('Prefill Time vs Prefill Length (Batch=1)')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'exp2_prefill_time.png'))
plt.close()

# Plot 2b: Area chart breakdown of prefill phase (per-operation, sum across all layers)
prefill_ops = [c for c in exp2.columns if c.startswith('prefill_') and not any(
    c.endswith(suffix) for suffix in ['_time', '_total', '_fine', '_len', '_embedding', '_logits', '_layers']) and c != 'prefill_len']
if prefill_ops:
    op_totals = {op: exp2[op].sum() for op in prefill_ops}
    top5_ops = sorted(op_totals, key=op_totals.get, reverse=True)[:5]
    other_ops = [op for op in prefill_ops if op not in top5_ops]
    top5_ops_sorted = sorted(top5_ops, key=lambda op: op_totals[op], reverse=True)
    stack_data = [exp2[op] for op in top5_ops_sorted]
    if other_ops:
        other_sum = exp2[other_ops].sum(axis=1)
        stack_data.append(other_sum)
        labels = [op.replace('prefill_', '') for op in top5_ops_sorted] + ['Other']
    else:
        labels = [op.replace('prefill_', '') for op in top5_ops_sorted]
    n = len(stack_data)
    cmap = plt.cm.get_cmap('viridis', n)
    colors = [cmap(i) for i in range(n)]
    plt.figure(figsize=(10, 6))
    stack = plt.stackplot(
        np.log2(exp2['prefill_len']),
        stack_data,
        labels=labels,
        alpha=1.0,
        colors=colors
    )
    plt.xlabel('log2(Prefill Length)')
    plt.ylabel('Time (s)')
    plt.title('Prefill Phase Operation Breakdown (Batch=1)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp2_prefill_breakdown_area.png'))
    plt.close()

# 3. exp3.csv: batch_size, prefill_time, decode_time, total_time, throughput, ...
exp3 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'profile_results/exp3.csv'))

# Plot 3a: End-to-end time and throughput vs log2(batch_size)
fig, ax1 = plt.subplots(figsize=(8, 5))
color = 'tab:blue'
ax1.set_xlabel('log2(Batch Size)')
ax1.set_ylabel('End-to-End Time (s)', color=color)
ax1.plot(np.log2(exp3['batch_size']), exp3['total_time'], 'o-', label='End-to-End Time', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Throughput (tokens/s)', color=color)
ax2.plot(np.log2(exp3['batch_size']), exp3['throughput'], 's--', label='Throughput', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.suptitle('End-to-End Time and Throughput vs Batch Size (Prefill=128, Decode=128)')
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(output_dir, 'exp3_end_to_end_time_and_throughput.png'))
plt.close()

print("Plots saved in 'plots' directory:")
# print("exp1_end_to_end_time.png, exp1_decode_breakdown_area.png, exp1_last_decode_breakdown_area.png, exp2_prefill_time.png, exp2_prefill_breakdown_area.png, exp3_end_to_end_time_and_throughput.png") 