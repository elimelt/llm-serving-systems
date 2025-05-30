import os
import csv
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
SPLIT_K_SLICES = [1, 2, 4, 8]

# Helper to parse shape from filename
def parse_shape_from_filename(filename):
    # e.g., gemm_1024_4096.gemm.csv -> (1024, 4096)
    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) >= 3:
        m = int(parts[1])
        n = int(parts[2].split('.')[0])
        return (m, n)
    return None

def get_csv_files(results_dir):
    return [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith('.csv')]

def main():
    best_performance = defaultdict(lambda: defaultdict(dict))  # shape -> split_k -> dict
    all_stats = defaultdict(lambda: defaultdict(list))

    csv_files = get_csv_files(RESULTS_DIR)
    for csv_file in csv_files:
        shape = parse_shape_from_filename(csv_file)
        if not shape:
            continue
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Provider'] != 'CUTLASS':
                    continue
                if row['split_k_mode'] != 'serial':
                    continue
                try:
                    split_k = int(row['split_k_slices'])
                    gflops = float(row['GFLOPs'])
                except Exception:
                    continue
                if split_k not in SPLIT_K_SLICES:
                    continue
                # Only consider successful runs
                if row['Status'] != 'success':
                    continue
                # Save all stats for later analysis
                all_stats[shape][split_k].append(row)
                # Update best performance
                if ('GFLOPs' not in best_performance[shape][split_k]) or (gflops > best_performance[shape][split_k]['GFLOPs']):
                    best_performance[shape][split_k] = {
                        'GFLOPs': gflops,
                        'Operation': row['Operation'],
                        'cta_m': row['cta_m'],
                        'cta_n': row['cta_n'],
                        'cta_k': row['cta_k'],
                        'kernel': row['Operation'],
                        'tile_size': (row['cta_m'], row['cta_n'], row['cta_k']),
                        'runtime': row['Runtime'],
                        'm': row['m'],
                        'n': row['n'],
                        'k': row['k'],
                        'split_k_slices': split_k,
                    }

    # Print summary
    print("\nCUTLASS GEMM Best Performance Summary (GFLOPs):")
    for shape in sorted(best_performance.keys()):
        print(f"\nGEMM shape (m, n): {shape}")
        for split_k in SPLIT_K_SLICES:
            if split_k in best_performance[shape]:
                stat = best_performance[shape][split_k]
                print(f"  split_k_slices={split_k}: {stat['GFLOPs']:.2f} GFLOPs | kernel={stat['kernel']} | tile={stat['tile_size']} | runtime={stat['runtime']} s")
            else:
                print(f"  split_k_slices={split_k}: No data")

    # Investigate best kernel
    print("\n--- Kernel Analysis ---")
    for shape in sorted(best_performance.keys()):
        print(f"\nGEMM shape (m, n): {shape}")
        for split_k in SPLIT_K_SLICES:
            if split_k in best_performance[shape]:
                stat = best_performance[shape][split_k]
                print(f"  split_k_slices={split_k}: kernel={stat['kernel']} | tile={stat['tile_size']} | GFLOPs={stat['GFLOPs']:.2f}")

    # Find the single best-performing kernel across all shapes and split_k_slices
    global_best = None
    for shape in best_performance:
        for split_k in best_performance[shape]:
            stat = best_performance[shape][split_k]
            if (global_best is None) or (stat['GFLOPs'] > global_best['GFLOPs']):
                global_best = {
                    'shape': shape,
                    'split_k': split_k,
                    **stat
                }

    print("\n--- Global Best CUTLASS Kernel ---")
    if global_best:
        print(f"Best kernel: {global_best['kernel']}")
        print(f"  Problem size (m, n, k): ({global_best['m']}, {global_best['n']}, {global_best['k']})")
        print(f"  Tile size (cta_m, cta_n, cta_k): {global_best['tile_size']}")
        print(f"  split_k_slices: {global_best['split_k']}")
        print(f"  GFLOPs: {global_best['GFLOPs']:.2f}")
        print(f"  Runtime: {global_best['runtime']} s")

    # Print a table of all best kernels for each shape/split_k
    print("\n--- Best Kernel for Each Shape and split_k_slices ---")
    print(f"{'Shape':>15} {'split_k':>8} {'GFLOPs':>10} {'Tile Size':>20} {'Kernel':>40}")
    for shape in sorted(best_performance.keys()):
        for split_k in SPLIT_K_SLICES:
            if split_k in best_performance[shape]:
                stat = best_performance[shape][split_k]
                print(f"{str(shape):>15} {split_k:>8} {stat['GFLOPs']:>10.2f} {str(stat['tile_size']):>20} {stat['kernel'][:40]:>40}")

if __name__ == '__main__':
    main()
