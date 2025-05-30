from continous_engine import Engine, Request
import torch

class InputRequest:
    def __init__(self, input_str: str, output_len: int):
        self.input_str = input_str
        self.output_len = output_len
        
class Scheduler:
    def __init__(self, engine: Engine, req_batch_size: int):
        self.engine = engine
        self.req_batch_size = req_batch_size
        self.pending_input_req: list[InputRequest] = []
        self.decode_req: list[Request] = []
        self.scheduled_prefill_req: list[Request] = []
        self.completed: list[Request] = []
        self.unique_req_id: int = 0
    
    def add_req(self, input_req: InputRequest):
        self.pending_input_req.append(input_req)
        
    def finished(self) -> bool:
        return not self.pending_input_req and not self.decode_req

    def get_req_batch_size(self) -> int:
        return len(self.decode_req) + len(self.scheduled_prefill_req)

    def run(self):
        # Schedule new prefill requests until batch is full or no pending inputs
        while self.get_req_batch_size() < self.req_batch_size and self.pending_input_req:
            input_req = self.pending_input_req.pop(0)
            self.scheduled_prefill_req.append(Request(
                req_id=self.unique_req_id,
                prompt_ids=self.engine.tokenizer.encode(input_req.input_str, return_tensors="pt")[0],
                target_len=input_req.output_len
            ))
            self.unique_req_id += 1

        # Build the list of requests to send to the engine
        request_list_total = []
        decode_num = 0
        for req in self.decode_req:
            request_list_total.append(req)
            decode_num += 1
        for req in self.scheduled_prefill_req:
            request_list_total.append(req)
        
        new_tokens = self.engine.run(request_list_total, decode_num)

        # Append newly generated tokens to each request's output buffer
        for new_tok, req in zip(new_tokens, request_list_total):
            req.output_token_ids = torch.cat(
                [req.output_token_ids, new_tok.unsqueeze(0)], dim=0
            )

        # Check which decode requests have finished and remove from the queue
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
            if self.scheduled_prefill_req[0].current_length >= self.scheduled_prefill_req[0].output_length:
                self.completed.append(self.scheduled_prefill_req.pop(0))
            else:
                self.decode_req.append(self.scheduled_prefill_req.pop(0))
    
    def print_completed(self):
        for i, req in enumerate(self.completed):
            text = self.engine.tokenizer.decode(
                req.output_token_ids, skip_special_tokens=True
            )
            print(f"Id = {i}: {text}")
