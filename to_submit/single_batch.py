import torch
from transformers import AutoTokenizer
import sys
sys.path.append("../")  # Adjust the path to import the helper module
from helper import WeightManager, apply_rope, extract_model_weights


class Engine:
    """
    A class to manage the generation engine.
    """
    def __init__(self):
        ########################################
        # Model Configuration Parameters
        ########################################
        self.weight_path = "/model/Meta-Llama-3-8B-Instruct"
        self.head_dim = 128         # Dimensionality of each attention head
        self.num_qo_heads = 32      # Total number of query/output heads
        self.num_kv_heads = 8       # Total number of key/value heads
        self.layers = 32            # Number of transformer layers

        # Load the tokenizer for text processing
        self.tokenizer = AutoTokenizer.from_pretrained("/model/Meta-Llama-3-8B-Instruct")

        # Initialize and load model weights using the helper module
        weight_manager = WeightManager()
        weight_manager.load_from_safe_tensor(self.weight_path)

        # Extract all required model weights from the weight_map
        self.weights = extract_model_weights(weight_manager.weight_map, self.layers)
        
        # Initialize KV cache and sequence tracking
        self.kv_cache = {}
    
    def run(self, input_ids, prefill=True):
        # manage kv cache and position tracking    
        if prefill:
            self.kv_cache.clear()

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.int32, device="cuda")
        else:  # already a tensor
            input_ids = input_ids.to(dtype=torch.int32, device="cuda")

        seq_len = input_ids.shape[0]
        hidden_state = self.weights["embedding"][input_ids]

        # Transformer layers
        for layer_idx in range(self.layers):

            # Layer-norm (RMS)
            rms = torch.sqrt(hidden_state.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
            x_norm = (hidden_state / rms).to(torch.float16) * self.weights["layernormAttn_weight"][layer_idx]

            # Projections
            k = x_norm @ self.weights["self_attn_k_proj_weight"][layer_idx].T
            v = x_norm @ self.weights["self_attn_v_proj_weight"][layer_idx].T
            q = x_norm @ self.weights["self_attn_q_proj_weight"][layer_idx].T

            # rope (position-aware)
            past_k = None
            past_v = None
            if layer_idx in self.kv_cache:
                past_k = self.kv_cache[layer_idx]["k"]
                past_v = self.kv_cache[layer_idx]["v"]

            past_len = 0 if past_k is None else past_k.shape[0]
            apply_rope(q, q, self.head_dim, offset=past_len)
            apply_rope(k, k, self.head_dim, offset=past_len)

            # Reshape / append to cache
            k = k.view(seq_len, self.num_kv_heads, self.head_dim)
            v = v.view(seq_len, self.num_kv_heads, self.head_dim)
            
           
            if past_k is None:
                k_total = k
                v_total = v
            else:
                k_total = torch.cat([past_k, k], dim=0)
                v_total = torch.cat([past_v, v], dim=0)

            # save fp16 copies to avoid autograd / keep memory low
            self.kv_cache[layer_idx] = {
                "k": k_total.detach().to(torch.float16),
                "v": v_total.detach().to(torch.float16),
            }

            # Attention (use only last token's q, all keys/values)
            group_size = self.num_qo_heads // self.num_kv_heads
            q_h = q.view(seq_len, self.num_qo_heads, self.head_dim)
            k_h = k_total.repeat_interleave(group_size, dim=1)
            v_h = v_total.repeat_interleave(group_size, dim=1)

            q_h = q_h.permute(1, 0, 2)
            k_h = k_h.permute(1, 0, 2)
            v_h = v_h.permute(1, 0, 2)

            scale = 1.0 / (self.head_dim ** 0.5)
            scores = (q_h @ k_h.transpose(-2, -1)) * scale

            # causal mask: new token(s) only attend to <= current position
            if seq_len == 1:
                pass
            else:
                causal = torch.tril(torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device))
                scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))

            attn = torch.softmax(scores, dim=-1)
            context = attn @ v_h
            context = context.permute(1, 0, 2).reshape(seq_len, -1)

            hidden_state = (context @ self.weights["o_proj_weight"][layer_idx].T) + hidden_state

            # Feed-forward network
            rms = torch.sqrt(hidden_state.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
            x_norm = (hidden_state / rms).to(torch.float16) * self.weights["layernormFFN_weight"][layer_idx]

            up = x_norm @ self.weights["up_proj_weight"][layer_idx].T
            gate = x_norm @ self.weights["gate_proj_weight"][layer_idx].T
            ff = (up * torch.nn.functional.silu(gate)) @ self.weights["down_proj_weight"][layer_idx].T

            hidden_state = ff + hidden_state

        # Final projection -> logits -> greedy pick
        rms = torch.sqrt(hidden_state.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
        norm = (hidden_state / rms).to(torch.float16) * self.weights["model_layernorm_weight"]
        logits = norm @ self.weights["lm_head_weight"].T

        next_token = logits.argmax(dim=-1)[-1].item()  # last position
        return next_token

    def generate(self, input_string, rounds=20):
        """
        Generate text from an input string
        
        Args:
            input_string: The input prompt
            rounds: Number of tokens to generate
            
        Returns:
            The generated text
        """
        # Encode the input string to token IDs
        input_ids = self.tokenizer.encode(input_string)
        
        # Initialize output with input tokens
        output_ids = input_ids.copy()

        new_token = self.run(output_ids)
        output_ids.append(new_token)
        
        # Generate additional tokens
        for round in range(rounds - 1):
            new_token = self.run(output_ids[-1:], prefill=False)
            output_ids.append(new_token)
        
        # Decode the output tokens to text
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return output_text

########################################
# Main Loop: Text Generation
########################################
if __name__ == "__main__":
    input_string = "Hi, who are you?"
    engine = Engine()
    output_text, t = engine.generate(input_string, rounds=20)
    print(f"Generated Text: {output_text}, time: {t}s")