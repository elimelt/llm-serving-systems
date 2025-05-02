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
        self.head_dim = 128  # Dimensionality of each attention head
        self.num_qo_heads = 32  # Total number of query/output heads
        self.num_kv_heads = 8  # Total number of key/value heads
        self.layers = 32  # Number of transformer layers

        # Load the tokenizer for text processing
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/model/Meta-Llama-3-8B-Instruct"
        )

        # Initialize and load model weights using the helper module
        weight_manager = WeightManager()
        weight_manager.load_from_safe_tensor(self.weight_path)

        # Extract all required model weights from the weight_map
        self.weights = extract_model_weights(weight_manager.weight_map, self.layers)

        self.kv_cache = {}

    @torch.inference_mode()
    def run(self, input_ids, *, prefill: bool = True) -> torch.Tensor:
        """
        parameters:
        input_ids: token ids (b, s) for prefill or (b, 1) for decode
        prefill: whether this is first pass or incremental step

        returns:
        next token ids (b, 1)
        """
        # reset kv cache
        if prefill:
            self.kv_cache.clear()

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.int32, device="cuda")
        else:
            input_ids = input_ids.to(dtype=torch.int32, device="cuda")

        B, S = input_ids.shape
        H_kv = self.num_kv_heads
        H_q = self.num_qo_heads
        D_h = self.head_dim
        group_size = H_q // H_kv
        scale = 1.0 / (D_h**0.5)

        # embedding lookup
        hidden_state = self.weights["embedding"][input_ids]

        # transformer
        for layer in range(self.layers):
            # layer norm
            rms = torch.sqrt(hidden_state.pow(2).mean(-1, keepdim=True) + 1e-5)
            x_norm = (hidden_state / rms).to(torch.float16) * self.weights["layernormAttn_weight"][layer]

            # projections
            d_model = x_norm.shape[-1]
            x_flat = x_norm.view(-1, d_model)

            k = x_flat @ self.weights["self_attn_k_proj_weight"][layer].T
            v = x_flat @ self.weights["self_attn_v_proj_weight"][layer].T
            q = x_flat @ self.weights["self_attn_q_proj_weight"][layer].T

            k = k.view(B, S, H_kv, D_h)
            v = v.view(B, S, H_kv, D_h)
            q = q.view(B, S, H_q, D_h)

            # rope
            past_len = 0
            if layer in self.kv_cache:
                past_len = self.kv_cache[layer]["k"].shape[1]

            for batch in range(B):
                apply_rope(
                    q[batch].view(S, -1),
                    output=q[batch].view(S, -1),
                    head_dim=D_h,
                    offset=past_len,
                )
                apply_rope(
                    k[batch].view(S, -1),
                    output=k[batch].view(S, -1),
                    head_dim=D_h,
                    offset=past_len,
                )

            # kv cache update
            if layer not in self.kv_cache:
                self.kv_cache[layer] = {
                    "k": k.to(torch.float16).clone(),
                    "v": v.to(torch.float16).clone(),
                }
            else:
                self.kv_cache[layer]["k"] = torch.cat(
                    [self.kv_cache[layer]["k"], k.to(torch.float16)], dim=1
                )
                self.kv_cache[layer]["v"] = torch.cat(
                    [self.kv_cache[layer]["v"], v.to(torch.float16)], dim=1
                )

            k_total = self.kv_cache[layer]["k"]
            v_total = self.kv_cache[layer]["v"]

            # repeat kv heads to match query heads
            k_rep = k_total.repeat_interleave(group_size, dim=2)
            v_rep = v_total.repeat_interleave(group_size, dim=2)

            # prepare for batch matrix multiplication
            q_t = q.permute(0, 2, 1, 3)
            k_t = k_rep.permute(0, 2, 1, 3)
            v_t = v_rep.permute(0, 2, 1, 3)

            # attention
            attn_scores = (q_t @ k_t.transpose(-2, -1)) * scale

            if prefill:
                causal = torch.tril(
                    torch.ones(
                        S, S + past_len, dtype=torch.bool, device=attn_scores.device
                    )
                )
                attn_scores = attn_scores.masked_fill(
                    (~causal).unsqueeze(0).unsqueeze(0), float("-inf")
                )

            attn_weights = torch.softmax(attn_scores, dim=-1)
            context = attn_weights @ v_t

            # merge heads and apply output projection
            context = context.permute(0, 2, 1, 3).contiguous().view(B, S, -1)
            hidden_state = (context @ self.weights["o_proj_weight"][layer].T) + hidden_state

            # feed-forward
            rms = torch.sqrt(hidden_state.pow(2).mean(-1, keepdim=True) + 1e-5)
            x_norm = (hidden_state / rms).to(torch.float16) * self.weights["layernormFFN_weight"][layer]

            up = x_norm.view(-1, d_model) @ self.weights["up_proj_weight"][layer].T
            gate = x_norm.view(-1, d_model) @ self.weights["gate_proj_weight"][layer].T
            ff = (up * torch.nn.functional.silu(gate)) @ self.weights["down_proj_weight"][layer].T
            hidden_state = ff.view(B, S, -1) + hidden_state

        # final logits
        rms = torch.sqrt(hidden_state.pow(2).mean(-1, keepdim=True) + 1e-5)
        norm = (hidden_state / rms).to(torch.float16) * self.weights["model_layernorm_weight"]
        logits = norm @ self.weights["lm_head_weight"].T

        next_tokens = logits.argmax(dim=-1)[:, -1:]
        return next_tokens.to(torch.int32).to(input_ids.device)

    def generate_batched(self, input_string_list, rounds=20):
        input_ids_list = self.tokenizer(
            input_string_list, return_tensors="pt", padding=False
        ).input_ids.to(device="cuda")

        output_ids_list = input_ids_list

        new_token = self.run(output_ids_list)
        output_ids_list = torch.cat((output_ids_list, new_token), dim=1)

        for round in range(rounds - 1):
            new_token = self.run(output_ids_list[:, -1:], prefill=False)
            output_ids_list = torch.cat((output_ids_list, new_token), dim=1)

        output_text = self.tokenizer.batch_decode(
            output_ids_list, skip_special_tokens=True
        )
        return output_text


########################################
# Main Loop: Text Generation
########################################
if __name__ == "__main__":
    input_string = "Hi, who are you?"
    batch_size = 64
    # input_string_list = [input_string for _ in range(10)]
    # another_input_string = "Hi, how are you?"
    # for _ in range(10):
    # input_string_list.append(another_input_string)
    # engine = Engine()
    # output_text = engine.generate_batched(input_string_list, rounds=20)
    # print("Generated Text:", output_text)
    engine = Engine()
    input_string_list = [input_string] * batch_size
    output_text = engine.generate_batched(input_string_list, rounds=22)
    print("Generated Text:", output_text)
