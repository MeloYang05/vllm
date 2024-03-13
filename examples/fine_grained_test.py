from vllm import LLM, SamplingParams
from vllm.sequence import Sequence, SequenceGroup
from typing import List
import time

def build_batched_token_ids(token_ids: List[int], batch_size: int) -> List[List[int]]:
    assert batch_size >= 1
    batched_token_ids: List[List[int]] = []
    for i in range(batch_size):
        batched_token_ids.append(token_ids[i:] + token_ids[:i])
    return batched_token_ids

block_size = 16
prefix_len = block_size * 32 + 1
gen_blocks = 64
gen_blocks_small = 16
requests_num = 50

llm = LLM(model="/mnt/nas/tao/models/Llama-2-7B-FP16", enable_prefix_caching=True, 
          block_size=block_size, max_num_seqs=requests_num)

prompt_token_ids = list(range(prefix_len))
batched_prompt_token_ids = build_batched_token_ids(prompt_token_ids, requests_num)

for prompt_token_ids in batched_prompt_token_ids:
    sampling_params = SamplingParams(temperature=0.0, max_tokens=block_size * gen_blocks, ignore_eos=True)
    req_id = str(next(llm.request_counter))
    arrival_time = time.monotonic()
    seq_id = next(llm.llm_engine.seq_counter)
    seq = Sequence(seq_id, "", prompt_token_ids, block_size)
    seq_group = SequenceGroup(req_id, [seq], sampling_params, arrival_time)
    llm.llm_engine.scheduler.add_seq_group(seq_group)

time1 = time.time()
outputs = []
# Running the first large program
step_outputs = llm.llm_engine.step()
for output in step_outputs:
    if output.finished:
        outputs.append(output)

# Add extra two short requests
req_ids = []
for i in range(2):
    prompt_token_ids = list(range(block_size * 16 * (i + 1), block_size * 16 * i, -1))
    sampling_params = SamplingParams(temperature=0.0, max_tokens=block_size * gen_blocks_small, ignore_eos=True)
    req_id = str(next(llm.request_counter))
    req_ids.append(req_id)
    arrival_time = time.monotonic()
    seq_id = next(llm.llm_engine.seq_counter)
    seq = Sequence(seq_id, "", prompt_token_ids, block_size)
    seq_group = SequenceGroup(req_id, [seq], sampling_params, arrival_time)
    llm.llm_engine.scheduler.add_seq_group(seq_group)

time3 = time.time()

while llm.llm_engine.has_unfinished_requests():
    step_outputs = llm.llm_engine.step()
    for output in step_outputs:
        if output.finished:
            if output.request_id in req_ids:
                print("Finish one short request")
                time4 = time.time()
                duration = time4 - time3
                print(f"Short Requsts Duration: {duration}s")
            outputs.append(output)
time2 = time.time()
duaration = time2 - time1
print(f"Eplased: {duaration}s")
