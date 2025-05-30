from chunked_engine import Engine, Request
import torch


class InputRequest:
    def __init__(self, input_str: str, output_len: int):
        self.input_str = input_str
        self.output_len = output_len


class Scheduler:
    def __init__(self, engine: Engine, token_batch_size: int):
        self.engine = engine
        self.token_batch_size = token_batch_size
        self.pending_input_req: list[InputRequest] = []
        self.decode_req: list[Request] = []
        self.scheduled_prefill_req: list[Request] = []
        self.completed: list[Request] = []
        self.unique_req_id: int = 0
        self.pending_prefill: list[Request] = []
        self.chunk_size = 128 # tested 128, 256, 512 + 128 consistently has better end->end latency

    def add_req(self, input_req: InputRequest):
        self.pending_input_req.append(input_req)

    def finished(self) -> bool:
        return (
            not self.pending_input_req
            and not self.decode_req
            and not self.pending_prefill
        )

    def get_token_batch_size(self) -> int:
        sum = 0
        for req in self.scheduled_prefill_req:
            sum += req.scheduling_length
        for req in self.decode_req:
            sum += 1
        return sum

    def run(self):
        # Schedule new prefill requests until batch is full or no pending inputs

        while self.pending_input_req:
            pending_req = self.pending_input_req.pop(0)
            req_id = self.unique_req_id
            self.unique_req_id += 1
            prompt_ids = self.engine.tokenizer(
                pending_req.input_str, return_tensors="pt"
            ).input_ids[0]
            req = Request(req_id, prompt_ids, pending_req.output_len)
            self.pending_prefill.append(req)

        current_budget_used = self.get_token_batch_size()
        available_budget = self.token_batch_size - current_budget_used

        # Schedule prefill request and move it to the scheduled_prefill_req.
        # Limit the budget, and chunk the request as necessary.
        # If a request is chunked, keep it still in the pending prefill queue. Otherwise, remove from the queue.
        # Set tokens of the current chunk in the request scheduling_pf_tokens and set remaining_prefill_tokens properly
        while available_budget > 0 and self.pending_prefill:
            req = self.pending_prefill.pop()
            chunk = min( self.chunk_size, available_budget, req.remaining_prefill_tokens)
            start = req.prompt_length - req.remaining_prefill_tokens
            req.scheduling_pf_tokens = req.prompt_token_ids[start : start + chunk]
            req.remaining_prefill_tokens -= chunk
            available_budget -= chunk
            if req.remaining_prefill_tokens == 0:
                req.last_chunk = True
            elif req.remaining_prefill_tokens > 0:
                self.pending_prefill.append(req)
            else:
                print("aaaaaaa")
            self.scheduled_prefill_req.append(req)
        # Build the list of requests to send to the engine
        request_list_total = []
        decode_num = 0
        for req in self.decode_req:
            request_list_total.append(req)
            decode_num += 1
        for req in self.scheduled_prefill_req:
            request_list_total.append(req)

        # Append newly generated tokens to each request's output buffer
        # For prefill, only append if this cycle is the last chunk
        new_tokens = self.engine.run(request_list_total, decode_num)
        for i in range(len(new_tokens)):
            if i < decode_num or request_list_total[i].last_chunk:
                request_list_total[i].output_token_ids = torch.cat(
                    [
                        request_list_total[i].output_token_ids,
                        new_tokens[i].unsqueeze(0),
                    ],
                    dim=0,
                )

        # Check which decode requests have finished
        ongoing_decode: list[Request] = []
        for i in range(decode_num):
            req = request_list_total[i]
            if req.current_length >= req.output_length:
                self.completed.append(req)
            else:
                ongoing_decode.append(req)
        self.decode_req = ongoing_decode

        # Move scheduled prefill requests into decode queue
        while self.scheduled_prefill_req:
            req = self.scheduled_prefill_req.pop(0)
            if req.last_chunk:
                if req.current_length >= req.output_length:
                    self.completed.append(req)
                else:
                    self.decode_req.append(req)

    def print_completed(self):
        for i, req in enumerate(self.completed):
            text = self.engine.tokenizer.decode(
                req.output_token_ids, skip_special_tokens=True
            )
            print(f"Id = {i}: {text}")
