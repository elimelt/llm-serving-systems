import time
import pandas as pd
import matplotlib.pyplot as plt
from no_kv import Engine as NoKVEngine
from single_batch import Engine as KVEngine

def get_input_str(length):
    """
    Generate a string of the specified length.
    """
    try:
        return open("input.txt", "r").read()[:length]
    except FileNotFoundError:
        return ''
    
def run_engine(engine, input_str, rounds=1):
    """
    Run the specified engine with the given input string.
    """
    output, t = engine.generate(input_str, rounds=rounds)
    return t, output


if __name__ == '__main__':
    kv_model = KVEngine()
    no_kv_model = NoKVEngine()
    
    SIZES = range(128, 2048, 128)
    res_kv = []
    res_no_kv = []
    in_str = get_input_str(1028)
    for sz in SIZES:
        print(f"input size: {sz}")
        
        t, output = run_engine(kv_model, in_str, rounds=sz)
        print(f"KVEngine: {t:.4f} seconds for size {sz} with kv")
        res_kv.append((t, output))
        t, output = run_engine(no_kv_model, in_str, rounds=sz)
        print(f"NoKVEngine: {t:.4f} seconds for size {sz} with no kv")
        res_no_kv.append((t, output))
        
        
    df_kv = pd.DataFrame(res_kv, columns=['Time', 'Output'])
    df_no_kv = pd.DataFrame(res_no_kv, columns=['Time', 'Output'])
    
    df_kv.to_csv('kv_engine_results.csv', index=False)
    df_no_kv.to_csv('no_kv_engine_results.csv', index=False)
    
    print("last output:", output)
    
    # Plotting the results
    times_kv = [r[0] for r in res_kv]
    times_no_kv = [r[0] for r in res_no_kv]
    plt.plot(SIZES, times_kv, label='KVEngine')
    plt.plot(SIZES, times_no_kv, label='NoKVEngine')
    
    plt.xlabel('Input Size')
    plt.ylabel('Time (seconds)')
    plt.title('KVEngine Performance')
    plt.savefig('kv_engine_performance.png')

    
        
    
    