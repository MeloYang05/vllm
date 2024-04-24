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
    print(f"seq_lens: {seq_lens} batch_size: {len(seq_lens)} elapsed: {elapsed}")


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


def decode_padding(llm: LLM, seq_lens: List[int], gen_len: int):
    prompt_token_ids = []
    for seq_len in seq_lens:
        prompt_token_id = list(range(seq_len))
        prompt_token_ids.append(prompt_token_id)
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
    print(f"seq_lens: {seq_lens} gen_len: {gen_len} decode_elapsed: {decode_elapsed}")


def prefill_profile(llm: LLM):
    # Warm up
    prefill(llm, 1024, 1)

    batch_sizes = [1, 2, 4]
    seq_lens = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    num_blocks = llm.llm_engine.scheduler.block_manager.gpu_allocator.num_blocks
    block_size = llm.llm_engine.scheduler.block_manager.gpu_allocator.block_size
    tokens_num_limit = num_blocks * block_size
    print(tokens_num_limit)
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            if seq_len * batch_size >= tokens_num_limit:
                continue
            prefill(llm, seq_len, batch_size)


def prefill_padding_profile(llm: LLM):
    prefill_padding(llm, [128, 128])
    prefill_padding(llm, [128, 512])
    prefill_padding(llm, [512, 512])
    prefill_padding(llm, [128, 1024])
    prefill_padding(llm, [1024, 1024])
    prefill_padding(llm, [128, 2048])
    prefill_padding(llm, [2048, 2048])
    prefill_padding(llm, [128, 4096])
    prefill_padding(llm, [4096, 4096])
    prefill_padding(llm, [128, 128, 128, 512])
    prefill_padding(llm, [512, 512, 512, 512])


def decode_profile(llm: LLM):
    # batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    batch_sizes = [64, 128, 256]
    seq_lens = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    num_blocks = llm.llm_engine.scheduler.block_manager.gpu_allocator.num_blocks
    block_size = llm.llm_engine.scheduler.block_manager.gpu_allocator.block_size
    tokens_num_limit = num_blocks * block_size
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            gen_len = max(min(seq_len / 4, 128), 32)
            if (seq_len + gen_len) * batch_size >= tokens_num_limit:
                continue
            decode(llm, seq_len, gen_len, batch_size)


def decode_padding_profile(llm):
    decode_padding(llm, [32] * 1 + [32] * 1, 128)
    decode_padding(llm, [32] * 1 + [1024] * 1, 128)
    decode_padding(llm, [1024] * 1 + [1024] * 1, 128)
    decode_padding(llm, [32] * 1 + [4096] * 1, 128)
    decode_padding(llm, [4096] * 1 + [4096] * 1, 128)


if __name__ == "__main__":
    llm = LLM(
        model="/mnt/nas/tao/models/Qwen2-72B-Chat_merged_int4v0.2",
        tensor_parallel_size=1,
        enforce_eager=True,
    )
    # prefill_profile(llm)
    # prefill_padding_profile(llm)
    # decode_profile(llm)
    # decode(llm, 64, 32, 128)
    decode_padding_profile(llm)
