from vllm import LLM, SamplingParams
from typing import List
import time


def prefill(llm: LLM, seq_len: int, batch_size: int):
    prompt_token_ids = [list(range(seq_len))] * batch_size
    sampling_params = SamplingParams(temperature=1, max_tokens=1)
    start_time = time.time()
    llm.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"seq_len: {seq_len} batch_size: {batch_size}, elapsed: {elapsed}")


def prefill_padding(llm: LLM, seq_lens: List[int]):
    prompt_token_ids = []
    for seq_len in seq_lens:
        prompt_token_id = list(range(seq_len))
        prompt_token_ids.append(prompt_token_id)
    sampling_params = SamplingParams(temperature=1, max_tokens=1)
    start_time = time.time()
    llm.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"seq_lens: {seq_lens} batch_size: {len(seq_lens)}, elapsed: {elapsed}")


def decode(llm: LLM, seq_len: int, gen_len: int, batch_size: int):
    prompt_token_ids = [list(range(seq_len))] * batch_size
    sampling_params = SamplingParams(temperature=1, max_tokens=1)
    prefill_start_time = time.time()
    llm.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    prefill_end_time = time.time()
    prefill_elapsed = prefill_end_time - prefill_start_time

    sampling_params = SamplingParams(temperature=1, max_tokens=gen_len, ignore_eos=True)
    start_time = time.time()
    outputs = llm.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    end_time = time.time()
    for output in outputs:
        assert len(output.outputs[0].token_ids) == gen_len
    elapsed = end_time - start_time
    decode_elapsed = elapsed - prefill_elapsed
    print(
        f"seq_len: {seq_len} batch_size: {batch_size} gen_len: {gen_len} decode_elapsed: {decode_elapsed}"
    )


def prefill_profile(llm: LLM):
    batch_sizes = [1, 2, 4]
    seq_lens = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    tokens_num_limit = 12352
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            if seq_len * batch_size >= tokens_num_limit:
                continue
            prefill(llm, seq_len, batch_size)


def prefill_padding_profile(llm: LLM):
    prefill_padding(llm, [128, 512])
    prefill_padding(llm, [128, 1024])
    prefill_padding(llm, [128, 2048])
    prefill_padding(llm, [128, 4096])


def decode_profile(llm: LLM):
    # batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    batch_sizes = [64, 128, 256]
    seq_lens = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    tokens_num_limit = 12352
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            gen_len = max(min(seq_len / 4, 128), 32)
            if (seq_len + gen_len) * batch_size >= tokens_num_limit:
                continue
            decode(llm, seq_len, gen_len, batch_size)


if __name__ == "__main__":
    llm = LLM(model="/mnt/nas/tao/models/Qwen2-72B-Chat_merged_int4v0.2")
    # prefill_profile(llm)
    # prefill_padding_profile(llm)
    decode_profile(llm)
    # decode(llm, 64, 32, 128)
