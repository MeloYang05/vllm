from vllm import LLM, SamplingParams
from typing import List
import time

def build_batched_token_ids(token_ids: List[int], batch_size: int) -> List[List[int]]:
    assert batch_size >= 1
    batched_token_ids: List[List[int]] = []
    for i in range(batch_size):
        batched_token_ids.append(token_ids[i:] + token_ids[:i])
    return batched_token_ids
        

prefix_len = 1025
requests_num = 48
block_size = 16
chunk_size = 1

llm = LLM(model="/mnt/nas/tao/models/Llama-2-7B-FP16", enable_prefix_caching=True, block_size=block_size)

prompt1_token_ids = list(range(prefix_len))
batched_prompt1_token_ids = build_batched_token_ids(prompt1_token_ids, requests_num)

sampling_params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
llm.generate(sampling_params=sampling_params, prompt_token_ids=batched_prompt1_token_ids, use_tqdm=False)

print("Start!")
sampling_params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
start_time = time.time()
llm.generate(sampling_params=sampling_params, prompt_token_ids=batched_prompt1_token_ids, use_tqdm=False)
end_time = time.time()
duaration = end_time - start_time
print(f"Eplapsed: {duaration}s")
