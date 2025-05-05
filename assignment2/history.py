from numpy import concat, require
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
        Parameters
        input_ids : List[torch.Tensor]
        prefill : bool
            Whether this is the first pass or a decode step.

        Returns
        torch.Tensor
            Next-token ids for every sequence in the same order as `input_ids`.
        """
        batch_size = len(input_ids)
        if prefill or not self.kv_cache:
            self.kv_cache = [
                [{} for _ in range(self.layers)] for _ in range(batch_size)
            ]

        # move to CUDA and build embedding
        hid_states: List[torch.Tensor] = []
        for i in range(batch_size):
            ids_i = input_ids[i].to(dtype=torch.int32, device="cuda")
            hid_states.append(self.weights["embedding"][ids_i])  # (S_i, d_model)

        # transformer layers
        group_size = self.num_qo_heads // self.num_kv_heads
        D_h = self.head_dim
        scale = 1.0 / (D_h**0.5)

        for layer in range(self.layers):
            # Loop over samples because sequence lengths differ
            for b in range(batch_size):
                x = hid_states[b]  # (T, d_model)
                rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
                x_norm = (x / rms).to(torch.float16) * self.weights[
                    "layernormAttn_weight"
                ][layer]

                # projections
                k = (
                    x_norm @ self.weights["self_attn_k_proj_weight"][layer].T
                )  # (T, d_model)
                v = x_norm @ self.weights["self_attn_v_proj_weight"][layer].T
                q = x_norm @ self.weights["self_attn_q_proj_weight"][layer].T

                # reshape for RoPE
                past_k = self.kv_cache[b][layer].get("k")  # none on first call
                past_len = 0 if past_k is None else past_k.shape[0]

                apply_rope(q, q, D_h, offset=past_len)
                apply_rope(k, k, D_h, offset=past_len)

                k = k.view(-1, self.num_kv_heads, D_h)  # (T, H_kv, d_h)
                v = v.view(-1, self.num_kv_heads, D_h)
                q = q.view(-1, self.num_qo_heads, D_h)  # (T, H_q , d_h)

                # update / build cache
                if past_k is None:
                    k_tot = k
                    v_tot = v
                else:
                    k_tot = torch.cat([past_k, k], dim=0)
                    v_tot = torch.cat([self.kv_cache[b][layer]["v"], v], dim=0)

                # keep fp16 cache
                self.kv_cache[b][layer]["k"] = k_tot.to(torch.float16)
                self.kv_cache[b][layer]["v"] = v_tot.to(torch.float16)

                # attention (only need last token's query when prefill==False)
                last_q = q[-1:] if not prefill else q  # (1 or T, H_q, d_h)
                k_rep = k_tot.repeat_interleave(group_size, dim=1)  # (T_tot, H_q, d_h)
                v_rep = v_tot.repeat_interleave(group_size, dim=1)

                q_t = last_q.permute(1, 0, 2)  # (H_q, L_q, d_h)
                k_t = k_rep.permute(1, 0, 2)  # (H_q, T_tot, d_h)
                scores = (q_t @ k_t.transpose(-2, -1)) * scale  # (H_q, L_q, T_tot)

                if prefill:
                    Lq, Ttot = scores.shape[-2:]
                    causal = torch.tril(
                        torch.ones(Lq, Ttot, dtype=torch.bool, device=scores.device)
                    )
                    scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))

                attn = torch.softmax(scores, dim=-1)
                v_t = v_rep.permute(1, 0, 2)  # (H_q, T_tot, d_h)
                ctx = attn @ v_t  # (H_q, L_q, d_h)
                ctx = ctx.permute(1, 0, 2).reshape(
                    last_q.shape[0], -1
                )  # (L_q, H_q*d_h)

                # residual + output projection
                y = ctx @ self.weights["o_proj_weight"][layer].T
                if prefill:
                    hid_states[b] = y + hid_states[b]
                else:  # replace only last token slice
                    hid_states[b] = torch.cat(
                        [hid_states[b][:-1], hid_states[b][-1:] + y], dim=0
                    )

                # feed-forward
                x = hid_states[b]
                rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
                x_norm = (x / rms).to(torch.float16) * self.weights[
                    "layernormFFN_weight"
                ][layer]
                up = x_norm @ self.weights["up_proj_weight"][layer].T
                gate = x_norm @ self.weights["gate_proj_weight"][layer].T
                ff = (up * torch.nn.functional.silu(gate)) @ self.weights[
                    "down_proj_weight"
                ][layer].T
                hid_states[b] = ff + x

        # final projection
        next_tokens = []
        for b in range(batch_size):
            x = hid_states[b]
            rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
            x_norm = (x / rms).to(torch.float16) * self.weights[
                "model_layernorm_weight"
            ]
            logits = x_norm @ self.weights["lm_head_weight"].T
            next_tokens.append(logits[-1].argmax().item())

        return torch.tensor(next_tokens, dtype=torch.int32).to(
            device=input_ids[0].device
        )

    def generate_batched(self, input_string, rounds=20):
        input_ids_list = []
        for input_string in input_string:
            input_ids = self.tokenizer(input_string, return_tensors="pt").input_ids[0]
            input_ids_list.append(input_ids)

        output_ids_list = input_ids_list
        new_token = self.run(input_ids_list)
        for i in range(len(input_ids_list)):
            output_ids_list[i] = torch.cat(
                (output_ids_list[i], new_token[i : i + 1]), dim=0
            )

        for round in range(rounds - 1):
            input_ids_list = []
            for output_ids in output_ids_list:
                input_ids_list.append(output_ids[-1:])
            new_token = self.run(input_ids_list, prefill=False)

            for i in range(len(input_ids_list)):
                output_ids_list[i] = torch.cat(
                    (output_ids_list[i], new_token[i : i + 1]), dim=0
                )
        output_text_list = []
        for output_ids in output_ids_list:
            output_text_list.append(
                self.tokenizer.decode(output_ids, skip_special_tokens=True)
            )
        return output_text_list


########################################
# Main Loop: Text Generation
########################################
if __name__ == "__main__":
    input_string = "Hi, who are you?"
    input_string_list = [input_string for _ in range(10)]
    another_input_string = "The University of Washington is located in"
    for _ in range(10):
        input_string_list.append(another_input_string)
    engine = Engine()
    output_text = engine.generate_batched(input_string_list, rounds=20)
    print("Generated Text:", output_text)
