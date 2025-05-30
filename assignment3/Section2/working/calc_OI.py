import numpy as np
from bench_helper import get_model_configs, BYTES_PER_ELEM

# Prefill and decode attention operational intensity formulas
# Prefill: OI = (H_qo * 4 * d * p + H_qo * 3 * p) / (b * (2 * d * H_qo + 2 * d * H_kv))
# Decode:  OI = (H_qo * c * (4 * d + 3)) / (b * d * (2 * H_qo + 2 * c * H_kv))

def prefill_OI(H_qo, H_kv, d, b, p):
    numerator = H_qo * 4 * d * p + H_qo * 3 * p
    denominator = b * (2 * d * H_qo + 2 * d * H_kv)
    return numerator / denominator

def decode_OI(H_qo, H_kv, d, b, c):
    numerator = H_qo * c * (4 * d + 3)
    denominator = b * d * (2 * H_qo + 2 * c * H_kv)
    return numerator / denominator

if __name__ == "__main__":
    model_configs = get_model_configs()
    p_values = [2 ** i for i in range(7, 16)]
    c_values = [2 ** i for i in range(7, 16)]
    b = BYTES_PER_ELEM
    
    # Create markdown table file
    with open("operational_intensity.md", "w") as md_file:
        md_file.write("# Operational Intensity Analysis\n\n")
        
        # Model configurations summary
        md_file.write("## Model Configurations\n\n")
        md_file.write("| Model | H_qo | H_kv | d |\n")
        md_file.write("|-------|-----:|-----:|--:|\n")
        for model_name, cfg in model_configs.items():
            md_file.write(f"| {model_name} | {cfg['num_qo_heads']} | {cfg['num_kv_heads']} | {cfg['head_dim']} |\n")
        md_file.write("\n")
        
        # PREFILL TABLE - all models in columns
        md_file.write("## Prefill Attention Operational Intensity\n\n")
        
        # Write header with model names
        header = "| Sequence Length (p) |"
        separator = "|-------------------:|"
        for model_name in model_configs.keys():
            header += f" {model_name} |"
            separator += "------------:|"
        md_file.write(f"{header}\n{separator}\n")
        
        # Write rows with OI values for each sequence length and each model
        for p in p_values:
            row = f"| {p:,} |"
            for model_name, cfg in model_configs.items():
                H_qo = cfg['num_qo_heads']
                H_kv = cfg['num_kv_heads']
                d = cfg['head_dim']
                oi = prefill_OI(H_qo, H_kv, d, b, p)
                row += f" {oi:.4f} |"
            md_file.write(f"{row}\n")
        
        # DECODE TABLE - all models in columns
        md_file.write("\n## Decode Attention Operational Intensity\n\n")
        
        # Write header with model names
        header = "| Context Length (c) |"
        separator = "|------------------:|"
        for model_name in model_configs.keys():
            header += f" {model_name} |"
            separator += "------------:|"
        md_file.write(f"{header}\n{separator}\n")
        
        # Write rows with OI values for each context length and each model
        for c in c_values:
            row = f"| {c:,} |"
            for model_name, cfg in model_configs.items():
                H_qo = cfg['num_qo_heads']
                H_kv = cfg['num_kv_heads']
                d = cfg['head_dim']
                oi = decode_OI(H_qo, H_kv, d, b, c)
                row += f" {oi:.6f} |"
            md_file.write(f"{row}\n")
    
    print("Markdown tables have been written to operational_intensity.md")
