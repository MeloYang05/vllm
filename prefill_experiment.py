import os
import openpyxl

workbook = openpyxl.Workbook()
sheet = workbook.active
sheet.append(["Prefix Length", "Gen Length", "Request Num", 
              "Prompt1 Prefill Batch Size", "Decoding Batch Size", "Prompt2 Prefill Batch Size", 
              "Prompt1 Prefill Latency(s)", "Decoding Latency/Token(ms)", 
              "Decoding Latency(s)", "Prompt2 Prefill Latency(s)", "Total Latency(s)"])

requests_nums = [1, 8, 32]
prefix_lens = [65, 257, 1025]
gen_lens = [64, 256, 512]

for requests_num in requests_nums:
    for prefix_len in prefix_lens:
        for gen_len in gen_lens:
            print(requests_num, prefix_len, gen_len)
            os.system("/home/yufan/vllm_src/bin/python examples/chunked_prefill.py --enable-prefix-caching " + 
                    f"--requests-num {requests_num} --prefix-len {prefix_len} --gen-len {gen_len} > 2.log")

            # status -1: Other Status
            # status 0: Normal Prefill
            # status 1: Decoding
            # status 2: Chunked Prefill
            status = -1
            prompt1_prefill_batch_sizes = []
            decode_batch_sizes = []
            prompt2_prefill_batch_sizes = []
            prompt1_prefill_time = 0
            decode_time = 0
            prompt2_prefill_time = 0
            total_time = 0
            with open("2.log") as log_file:
                for line in log_file:
                    line = line.strip()
                    if line[:len("Start!")] == "Start!":
                        status = 0
                    elif line[:len("Prompt1 Prefilling Elapased")] == "Prompt1 Prefilling Elapased":
                        status = 1
                        prompt1_prefill_time = float(line[len("Prompt1 Prefilling Elapased: "):-1])
                    elif line[:len("Decoding Elapased")] == "Decoding Elapased":
                        status = 2
                        decode_time = float(line[len("Decoding Elapased: "):-1])
                    elif line[:len("Prompt2 Prefilling Elapased")] == "Prompt2 Prefilling Elapased":
                        prompt2_prefill_time = float(line[len("Prompt2 Prefilling Elapased: "):-1])
                    elif line[:len("Elapsed time")] == "Elapsed time":
                        total_time = float(line[len("Elapsed time: "):-1])
                        status = -1
                    if status == 0 and line[:len("Normal Prefill batch size")] == "Normal Prefill batch size":
                        prompt1_prefill_batch_sizes.append(int(line[len("Normal Prefill batch size: "):]))
                    elif status == 1 and line[:len("Decode batch size")] == "Decode batch size":
                        decode_batch_sizes.append(int(line[len("Decode batch size: "):]))
                    elif status == 2 and line[:len("Normal Prefill batch size")] == "Normal Prefill batch size":
                        prompt2_prefill_batch_sizes.append(int(line[len("Normal Prefill batch size: "):]))

            prompt1_prefill_batch_size = sum(prompt1_prefill_batch_sizes) / len(prompt1_prefill_batch_sizes)
            decode_batch_size = sum(decode_batch_sizes) / len(decode_batch_sizes)
            prompt2_prefill_batch_size = sum(prompt2_prefill_batch_sizes) / len(prompt2_prefill_batch_sizes)

            print(prompt1_prefill_batch_size)
            print(decode_batch_size)
            print(prompt2_prefill_batch_size)
            print(prompt1_prefill_time)
            print(decode_time)
            print(prompt2_prefill_time)
            print(total_time)
            print()

            sheet.append([prefix_len, gen_len, requests_num,
                        prompt1_prefill_batch_size, decode_batch_size, prompt2_prefill_batch_size,
                        prompt1_prefill_time, decode_time * 1000 / gen_len,
                        decode_time, prompt2_prefill_time, total_time])

excel_file_path = 'prefill_statistics.xlsx'
workbook.save(excel_file_path)
workbook.close()

