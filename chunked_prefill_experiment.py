import os
import openpyxl

workbook = openpyxl.Workbook()
sheet = workbook.active
sheet.append(["Prefix Length", "Gen Length", "Chunked Size", "Chunk Num", "Request Num", 
              "Prefix Prefill Batch Size", "Decoding Batch Size", "Chunked Prefill Batch Size", 
              "Prefix Prefill Latency(s)", "Decoding Latency/Token(ms)", 
              "Decoding Latency(s)", "Chunked Prefill Latency/Chunk(ms)", 
              "Chunked Prefill Latency(s)", "Total Latency(s)"])

requests_nums = [1, 8, 32]
prefix_lens = [65, 257, 1025]
gen_lens = [64, 256, 512]
chunk_nums = [1, 4, 16]

for requests_num in requests_nums:
    for prefix_len in prefix_lens:
        for gen_len in gen_lens:
            for chunk_num in chunk_nums:
                chunked_size = int(gen_len / chunk_num)
                print(requests_num, prefix_len, gen_len, chunk_num)
                os.system("/home/yufan/vllm_src/bin/python examples/chunked_prefill.py --enable-chunked-prefill --enable-prefix-caching " + 
                        f"--requests-num {requests_num} --prefix-len {prefix_len} --gen-len {gen_len} --chunked-size {chunked_size} > 1.log")

                # status -1: Other Status
                # status 0: Normal Prefill
                # status 1: Decoding
                # status 2: Chunked Prefill
                status = -1
                normal_prefill_batch_sizes = []
                decode_batch_sizes = []
                chunked_prefill_batch_sizes = []
                prefix_prefill_time = 0
                decode_time = 0
                chunked_prefill_time = 0
                total_time = 0
                with open("1.log") as log_file:
                    for line in log_file:
                        line = line.strip()
                        if line[:len("Start!")] == "Start!":
                            status = 0
                        elif line[:len("Prefix Prefilling Elapased")] == "Prefix Prefilling Elapased":
                            status = 1
                            prefix_prefill_time = float(line[len("Prefix Prefilling Elapased: "):-1])
                        elif line[:len("Decoding Elapased")] == "Decoding Elapased":
                            status = 2
                            decode_time = float(line[len("Decoding Elapased: "):-1])
                        elif line[:len("Chunked Prefilling Elapased")] == "Chunked Prefilling Elapased":
                            chunked_prefill_time = float(line[len("Chunked Prefilling Elapased: "):-1])
                        elif line[:len("Elapsed time")] == "Elapsed time":
                            total_time = float(line[len("Elapsed time: "):-1])
                            status = -1
                        if status == 0 and line[:len("Normal Prefill batch size")] == "Normal Prefill batch size":
                            normal_prefill_batch_sizes.append(int(line[len("Normal Prefill batch size: "):]))
                        elif status == 1 and line[:len("Decode batch size")] == "Decode batch size":
                            decode_batch_sizes.append(int(line[len("Decode batch size: "):]))
                        elif status == 2 and line[:len("Chunked Prefill batch size")] == "Chunked Prefill batch size":
                            chunked_prefill_batch_sizes.append(int(line[len("Chunked Prefill batch size: "):]))

                normal_prefill_batch_size = sum(normal_prefill_batch_sizes) / len(normal_prefill_batch_sizes)
                decode_batch_size = sum(decode_batch_sizes) / len(decode_batch_sizes)
                chunked_prefill_batch_size = sum(chunked_prefill_batch_sizes) / len(chunked_prefill_batch_sizes)

                print(normal_prefill_batch_size)
                print(decode_batch_size)
                print(chunked_prefill_batch_size)
                print(prefix_prefill_time)
                print(decode_time)
                print(chunked_prefill_time)
                print(total_time)
                print()

                sheet.append([prefix_len, gen_len, chunked_size, chunk_num, requests_num,
                            normal_prefill_batch_size, decode_batch_size, chunked_prefill_batch_size,
                            prefix_prefill_time, decode_time * 1000 / gen_len,
                            decode_time, chunked_prefill_time * 1000 / chunk_num, chunked_prefill_time, 
                            total_time])

excel_file_path = 'chunked_prefill_statistics.xlsx'
workbook.save(excel_file_path)
workbook.close()

