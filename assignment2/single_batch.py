import torch
from transformers import AutoTokenizer
import sys
sys.path.append("../")  # Adjust the path to import the helper module
from helper import WeightManager, apply_rope, extract_model_weights
import time


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
        self.seq_len = 0
    
    def run(self, input_ids, prefill=True):
        # Clear the KV cache on prefill operations
        if prefill:
            self.kv_cache = {}
            self.seq_len = 0
            offset = 0
        else:
            offset = self.seq_len

        # Embed input tokens
        input_tensor = torch.tensor(input_ids, dtype=torch.int32, device='cuda')
        hidden_state = self.weights["embedding"][input_tensor]

        for current_layer in range(self.layers):
            # --- Self-Attention Block ---
            # RMSNorm before attention
            rms = torch.sqrt(torch.mean(hidden_state ** 2, dim=-1, keepdim=True) + 1e-5)
            normalized_x = hidden_state / rms
            x = normalized_x.to(torch.float16) * self.weights["layernormAttn_weight"][current_layer]

            # Project to query, key, and value
            q = x.matmul(self.weights["self_attn_q_proj_weight"][current_layer].t())
            k = x.matmul(self.weights["self_attn_k_proj_weight"][current_layer].t())
            v = x.matmul(self.weights["self_attn_v_proj_weight"][current_layer].t())

            # Apply RoPE to query and key vectors with proper position offsets
            # Each token position needs to be processed with its absolute position
            seq_offset = offset
            for i in range(len(input_ids)):
                # Process each token with its proper position
                token_pos = seq_offset + i
                if len(input_ids) > 1:  # If we're in prefill mode with multiple tokens
                    # Apply rope to slices for each token
                    apply_rope(q[i:i+1], output=q[i:i+1], head_dim=self.head_dim, offset=token_pos)
                    apply_rope(k[i:i+1], output=k[i:i+1], head_dim=self.head_dim, offset=token_pos)
                else:
                    # Single token (usually in decode mode)
                    apply_rope(q, output=q, head_dim=self.head_dim, offset=token_pos)
                    apply_rope(k, output=k, head_dim=self.head_dim, offset=token_pos)

            # Reshape QKV tensors for multi-head attention
            sub_q = q.view(-1, self.num_qo_heads, self.head_dim)
            sub_k = k.view(-1, self.num_kv_heads, self.head_dim)
            sub_v = v.view(-1, self.num_kv_heads, self.head_dim)

            # Key-Value cache management
            if prefill or (current_layer not in self.kv_cache):
                # Initialize cache for this layer
                self.kv_cache[current_layer] = {"k": sub_k, "v": sub_v}
            else:
                # Append new keys and values to the cache
                self.kv_cache[current_layer]["k"] = torch.cat([self.kv_cache[current_layer]["k"], sub_k], dim=0)
                self.kv_cache[current_layer]["v"] = torch.cat([self.kv_cache[current_layer]["v"], sub_v], dim=0)

            # Get the full cached keys and values
            cache_k = self.kv_cache[current_layer]["k"]
            cache_v = self.kv_cache[current_layer]["v"]

            # Group-query attention: repeat KV heads to match QO heads
            scale = 1.0 / (self.head_dim ** 0.5)
            group_size = self.num_qo_heads // self.num_kv_heads
            
            # Expand KV heads to match QO heads with repeat_interleave
            expanded_k = cache_k.repeat_interleave(group_size, dim=1)
            expanded_v = cache_v.repeat_interleave(group_size, dim=1)

            # Prepare for batch matrix multiplication (transpose dimensions)
            sub_q_t = sub_q.permute(1, 0, 2)  # [num_qo_heads, seq_q, head_dim]
            sub_k_t = expanded_k.permute(1, 0, 2)  # [num_qo_heads, seq_k, head_dim]
            sub_v_t = expanded_v.permute(1, 0, 2)  # [num_qo_heads, seq_v, head_dim]

            # Calculate attention scores
            scores = torch.matmul(sub_q_t, sub_k_t.transpose(-2, -1)) * scale

            # Create causal attention mask
            n_q = sub_q.shape[0]  # Current sequence length
            n_k = cache_k.shape[0]  # Full sequence length (with cache)
            
            # Create appropriate attention mask
            if not prefill:
                assert n_q == 1
                # For single-token generation, allow attention to all previous tokens
                causal_mask = torch.ones(n_q, n_k, dtype=torch.bool, device=scores.device)
            else:
                # For prefill or multi-token inputs, use standard causal mask
                causal_mask = torch.tril(torch.ones(n_q, n_k, dtype=torch.bool, device=scores.device))
            
            # Apply causal mask to attention scores
            scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

            # Apply softmax to get attention weights
            attn_weights = torch.softmax(scores, dim=-1)

            # Compute weighted sum of values
            attn_output = torch.matmul(attn_weights, sub_v_t)
            attn_output = attn_output.permute(1, 0, 2)  # [seq_q, num_qo_heads, head_dim]
            attn_output = attn_output.reshape(-1, self.num_qo_heads * self.head_dim)

            # Project attention output and add residual connection
            attn_out = attn_output.matmul(self.weights["o_proj_weight"][current_layer].t())
            hidden_state_after_attn = attn_out + hidden_state

            # --- Feed-Forward Network Block ---
            # RMSNorm before FFN
            rms = torch.sqrt(torch.mean(hidden_state_after_attn ** 2, dim=-1, keepdim=True) + 1e-5)
            normalized_x = hidden_state_after_attn / rms
            ffn_input = normalized_x.to(torch.float16) * self.weights["layernormFFN_weight"][current_layer]

            # SwiGLU activation
            up_proj = ffn_input.matmul(self.weights["up_proj_weight"][current_layer].t())
            gate_proj = ffn_input.matmul(self.weights["gate_proj_weight"][current_layer].t())
            activated = up_proj * torch.nn.functional.silu(gate_proj)
            
            # Project and add residual connection
            ffn_output = activated.matmul(self.weights["down_proj_weight"][current_layer].t())
            hidden_state = ffn_output + hidden_state_after_attn

        # --- Final Layer Norm and Logits ---
        # Final RMSNorm
        rms = torch.sqrt(torch.mean(hidden_state ** 2, dim=-1, keepdim=True) + 1e-5)
        normalized_x = hidden_state / rms
        final_output = normalized_x.to(torch.float16) * self.weights["model_layernorm_weight"]
        
        # Project to vocabulary logits
        logits = final_output.matmul(self.weights["lm_head_weight"].t())
        
        # Update sequence tracking
        self.seq_len += len(input_ids)
        
        # Select the most likely next token
        next_token = torch.argmax(logits[-1]).item()
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
        time_tst = 0
        # Initial prefill pass with the entire prompt
        start = time.perf_counter()
        new_token = self.run(output_ids, prefill=True)
        stop = time.perf_counter()
        time_tst += ( stop - start )
        output_ids.append(new_token)
        
        # Generate additional tokens
        for round in range(rounds - 1):
            # print(f"Round {round}")
            start = time.perf_counter()
            # Generate next token with proper context
            new_token = self.run([output_ids[-1]], prefill=False)
            stop = time.perf_counter()
            time_tst += ( stop - start )
            output_ids.append(new_token)
        
        # Decode the output tokens to text
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return output_text, time_tst


########################################
# Main Loop: Text Generation
########################################
if __name__ == "__main__":
    input_string = "Hi, who are you?"
    engine = Engine()
    output_text = engine.generate(input_string, rounds=128)
    output_text = engine.generate(input_string, rounds=256)
    output_text = engine.generate(input_string, rounds=512)
    output_text = engine.generate(input_string, rounds=640)
    output_text = engine.generate(input_string, rounds=768)
    output_text = engine.generate(input_string, rounds=896)
    print(f"Generated Text: {output_text}")
