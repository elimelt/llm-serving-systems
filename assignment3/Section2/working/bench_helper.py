import torch
from transformers import AutoConfig

MODEL_IDS = {
    'LLaMA2-7B': 'meta-llama/Llama-2-7b-hf',
    'LLaMA3-8B': 'meta-llama/Meta-Llama-3-8B',
    'LLaMA3-70B': 'meta-llama/Meta-Llama-3-70B',
}

DTYPE = torch.float16
DEVICE = 'cuda'
BYTES_PER_ELEM = 2

def get_model_configs():
    models = {}
    for name, model_id in MODEL_IDS.items():
        config = AutoConfig.from_pretrained(model_id)
        head_dim = getattr(config, 'head_dim', None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        models[name] = {
            'num_qo_heads': config.num_attention_heads,
            'num_kv_heads': getattr(config, 'num_key_value_heads', config.num_attention_heads),
            'head_dim': head_dim,
            'layers': config.num_hidden_layers,
        }
    return models 