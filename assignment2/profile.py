
import time
import matplotlib.pyplot as plt
from no_kv import Engine as NoKVEngine
from single_batch import Engine as KVEngine


def main():
    input_len = 1024
    output_lens = list( range( 128, 2048 + 1, 128 ) )
    output_lens = list( range( 128, 1024, 128 ) )
    print(output_lens)
    
    # input_str = get_input_str( input_len, print_text=False )
    input_str = "The university of washington"
    no_kv_engine = NoKVEngine()
    kv_engine = KVEngine()
    
    # print(f"measuring no-kv ")
    # time_nokv = measure( no_kv_engine, input_str, output_lens )
    # print(f"time: {str( time_nokv )}")
        
    print(f"measuring kv ")
    time_kv = measure( kv_engine, input_str, output_lens )
    print(f"time: {str( time_kv )}")
    
    

def measure( engine, input_str, output_lens ):
    input_ids = engine.tokenizer.encode(input_str)
    tmp = len(input_ids)
    print(f"input id len: {tmp}")
    times = []
    for length in output_lens:
        total_query_time = 0
        output_ids = input_ids.copy()
        start = time.perf_counter()
        new_token = engine.run( output_ids, prefill=True )
        stop = time.perf_counter()
        output_ids.append( new_token )
        
        total_query_time += ( stop - start )
        for round in range( length ):
            start = time.perf_counter()
            new_token = engine.run( output_ids, prefill=False )
            stop = time.perf_counter()
            output_ids.append( new_token )
            
            total_query_time += ( stop - start )
        print( total_query_time )
        times.append( total_query_time )
        output_text = engine.tokenizer.decode( output_ids, skip_special_tokens=True)
        print( output_text )
             
    return times



def get_input_str( input_len, print_text=False ):
    input_str_filepath = 'input.txt'
    with open( input_str_filepath, 'r', encoding='utf-8') as f:
        str = f.read().split()
    input_str = str[ :input_len - 206 ]
    
    text = ' '.join(input_str)

    if print_text:
        print( text )
        
    return text
    

if __name__ == "__main__":
    main()