from vllm import LLM, SamplingParams
from typing import List
import time

def build_batched_token_ids(token_ids: List[int], batch_size: int) -> List[List[int]]:
    assert batch_size >= 1
    batched_token_ids: List[List[int]] = []
    for i in range(batch_size):
        batched_token_ids.append(token_ids[i:] + token_ids[:i])
    return batched_token_ids

enable_chunked_prefill = False
enable_prefix_caching = True
batch_size = 25
gen_len = 512
chunked_size = 32
chunked_steps = int(gen_len / chunked_size)
prefix_len = 1001

prompt1 = (
    "You are a professional product manager, and you are skilled at writing "
    "clear and detailed product document. Your document will be sent to programmers "
    "to write codes to formulate the product you design. Those programmers hope "
    "that your document explains carefully the function of the modules they take "
    "responsibility of. Today you have discusses with the CTO in your company, and "
    "you decide to build a smart voice chat robot based on the emerging AIGC techniques "
    "Please write a product document, which contains: 1. the general purpose of the "
    "smart voice chat robot. 2. Divide the entire chat robot into several sub modules,"
    "and explain the function of each module. 3. Explain the technical requirement of "
    "each submodule, and explain to programmers about how they should write the code "
    "and finish the module"
    )

prompt2 = (
    "You are a professional software programer, and you are skilled at writing "
    "clear and detailed code to fullfill requirement in a product document,"
    "where the document is sent by a product manager to you. Your task is to meet "
    "every functions raised in the document. Besides, you are a good programmer so "
    "that you have good coding habits and quelity. You are proficient at writing python "
    "code, and your coding style is very clean and easy to follow. You have a good habit "
    "of writing comments to explain your code. Today the CTO of your company and "
    "a prodct manager have a meeting and they decide to build a smart voice chat robot "
    "based on the emerging AIGC techniques. Your task is to write the code of a module "
    "of the voice chat bot based on the product document provided by the product manager. "
    "Here is the document provided by the product manager:"
    )

for i in range(3):
    prompt1 += prompt1
    prompt2 += prompt2

llm = LLM(model="/mnt/nas/tao/models/Llama-2-7B-FP16", enable_prefix_caching=True, block_size=16)
# prompt1_token_ids = llm.llm_engine.tokenizer.encode(prompt1)
# prompt2_token_ids = llm.llm_engine.tokenizer.encode(prompt2)
# prompt1_token_ids = prompt1_token_ids[:prefix_len]
# prompt2_token_ids = prompt2_token_ids[:prefix_len]

prompt1_token_ids = list(range(prefix_len))
prompt2_token_ids = list(range(prefix_len, prefix_len * 2))

batched_prompt1_token_ids = build_batched_token_ids(prompt1_token_ids, batch_size)
batched_prompt2_token_ids = build_batched_token_ids(prompt2_token_ids, batch_size)

start_time = time.time()
if enable_chunked_prefill:
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)
    outputs = llm.generate(sampling_params=sampling_params, prompt_token_ids=batched_prompt1_token_ids + batched_prompt2_token_ids)
    prompt1_gen_token = outputs[0].outputs[0].token_ids[0]
    prompt1_token_ids.append(prompt1_gen_token)

    decode_start_time = time.time()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=gen_len)
    outputs = llm.generate(sampling_params=sampling_params, prompt_token_ids=batched_prompt1_token_ids)
    decode_end_time = time.time()
    decode_time = decode_end_time - decode_start_time
    print(f"Decoding Elapased: {decode_time}s")

    chunked_prefill_start_time = time.time()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)
    for i in range(chunked_steps):
        for j in range(batch_size):
            batched_prompt2_token_ids[j] += outputs[j].outputs[0].token_ids[i*chunked_size:(i+1)*chunked_size]
        llm.generate(sampling_params=sampling_params, prompt_token_ids=batched_prompt2_token_ids)
    chunked_prefill_end_time = time.time()
    chunked_prefill_time = chunked_prefill_end_time - chunked_prefill_start_time
    print(f"Chunked Prefilling Elapased: {chunked_prefill_time}s")
else:
    decode_start_time = time.time()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=gen_len)
    outputs = llm.generate(sampling_params=sampling_params, prompt_token_ids=batched_prompt1_token_ids)
    decode_end_time = time.time()
    decode_time = decode_end_time - decode_start_time
    print(f"Decoding Elapased: {decode_time}s")
    
    for i in range(batch_size):
        batched_prompt2_token_ids[i] += outputs[i].outputs[0].token_ids
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)
    llm.generate(sampling_params=sampling_params, prompt_token_ids=batched_prompt2_token_ids)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
    