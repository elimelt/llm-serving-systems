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
        self.weights = extract_model_weights(weight_manager.weight_map, self.layers)

        self.kv_cache = {}

    @torch.inference_mode()
    def run(
        self, input_ids, *, attention_mask=None, prefill: bool = True
    ) -> torch.Tensor:
        if prefill:
            self.kv_cache.clear()

        # move inputs to device
        input_ids = input_ids.to(dtype=torch.int32, device="cuda")
        B, S = input_ids.shape

        # create default attention mask if none provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device="cuda")
        else:
            attention_mask = attention_mask.to(dtype=torch.bool, device="cuda")

        H_kv = self.num_kv_heads
        H_q = self.num_qo_heads
        D_h = self.head_dim
        group_size = H_q // H_kv
        scale = 1.0 / (D_h**0.5)

        hidden_state = self.weights["embedding"][input_ids]

        for layer in range(self.layers):
            # normalize
            rms = torch.sqrt(hidden_state.pow(2).mean(-1, keepdim=True) + 1e-5)
            x_norm = (hidden_state / rms).to(torch.float16) * self.weights[
                "layernormAttn_weight"
            ][layer]
            d_model = x_norm.shape[-1]
            x_flat = x_norm.view(-1, d_model)

            # compute qkv
            k = x_flat @ self.weights["self_attn_k_proj_weight"][layer].T
            v = x_flat @ self.weights["self_attn_v_proj_weight"][layer].T
            q = x_flat @ self.weights["self_attn_q_proj_weight"][layer].T

            k = k.view(B, S, H_kv, D_h)
            v = v.view(B, S, H_kv, D_h)
            q = q.view(B, S, H_q, D_h)

            past_len = 0
            if layer in self.kv_cache:
                past_len = self.kv_cache[layer]["k"].shape[1]

            # rope
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

            # update or initialize kv cache
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

            # grouped cross-attention prep
            k_rep = k_total.repeat_interleave(group_size, dim=2)
            v_rep = v_total.repeat_interleave(group_size, dim=2)

            q_t = q.permute(0, 2, 1, 3)
            k_t = k_rep.permute(0, 2, 1, 3)
            v_t = v_rep.permute(0, 2, 1, 3)

            # raw attention scores
            attn_scores = (q_t @ k_t.transpose(-2, -1)) * scale

            # --- unified mask: causal + padding ---
            B2, H2, S2, T = attn_scores.shape
            device = attn_scores.device
            # causal mask with shift for past_len
            causal = torch.tril(
                torch.ones((S2, T), dtype=torch.bool, device=device), diagonal=past_len
            )
            causal = causal.unsqueeze(0).unsqueeze(0).expand(B2, H2, S2, T)
            # padding mask: prepend ones for cached, then use attention_mask
            valid = torch.cat(
                [
                    torch.ones(
                        (B2, T - attention_mask.size(1)),
                        dtype=torch.bool,
                        device=device,
                    ),
                    attention_mask,
                ],
                dim=1,
            )
            padding = valid.unsqueeze(1).unsqueeze(2).expand(B2, H2, S2, T)
            mask = causal & padding
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

            attn_weights = torch.softmax(attn_scores, dim=-1)
            context = attn_weights @ v_t
            context = context.permute(0, 2, 1, 3).contiguous().view(B, S, -1)

            # projection + residual
            hidden_state = (
                context @ self.weights["o_proj_weight"][layer].T
            ) + hidden_state

            # -- FFN block --
            rms = torch.sqrt(hidden_state.pow(2).mean(-1, keepdim=True) + 1e-5)
            x_norm = (hidden_state / rms).to(torch.float16) * self.weights[
                "layernormFFN_weight"
            ][layer]
            up = x_norm.view(-1, d_model) @ self.weights["up_proj_weight"][layer].T
            gate = x_norm.view(-1, d_model) @ self.weights["gate_proj_weight"][layer].T
            ff = (up * torch.nn.functional.silu(gate)) @ self.weights[
                "down_proj_weight"
            ][layer].T
            hidden_state = ff.view(B, S, -1) + hidden_state

        # final norm + vocab projection
        rms = torch.sqrt(hidden_state.pow(2).mean(-1, keepdim=True) + 1e-5)
        norm = (hidden_state / rms).to(torch.float16) * self.weights[
            "model_layernorm_weight"
        ]
        logits = norm @ self.weights["lm_head_weight"].T

        # select next token from last valid position
        last_indices = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(B, device="cuda")
        next_tokens = logits[batch_idx, last_indices].argmax(dim=-1, keepdim=True)

        return next_tokens.to(torch.int32).to(input_ids.device)

    def generate_batched(self, input_string_list, rounds=20):
        encoded = self.tokenizer(
            input_string_list,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids = encoded.input_ids.to("cuda")
        attention_mask = encoded.attention_mask.to("cuda")

        output_ids = input_ids
        # prefill
        new_token = self.run(output_ids, attention_mask=attention_mask, prefill=True)
        output_ids = torch.cat((output_ids, new_token), dim=1)
        # unmask new token
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), device="cuda"),
            ],
            dim=1,
        )

        for _ in range(rounds - 1):
            new_token = self.run(output_ids[:, -1:], prefill=False)
            output_ids = torch.cat((output_ids, new_token), dim=1)

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)


if __name__ == "__main__":
    batch_size = 2
    input_string = "Hi, who are you?"
    input_string_list = [input_string] * (batch_size // 2) + ["Hi, ?"] * (
        batch_size // 2
    )
    engine = Engine()
    output_text = engine.generate_batched(input_string_list, rounds=22)
    print("Generated Text:", output_text)
