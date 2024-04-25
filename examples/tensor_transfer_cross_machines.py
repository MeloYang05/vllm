import torch
import torch.distributed as dist
import argparse
import time
from typing import Tuple


def send_revc_tensor(shape) -> Tuple[torch.Tensor, float]:
    # Create some sample data on the source GPU
    tensor = torch.randn(shape).cuda()

    # Synchronize source GPU before copying
    torch.cuda.synchronize()

    start_time = time.time()
    # Send data from source GPU to destination GPU using NCCL
    if dist.get_rank() == 0:  # Assuming process with rank 0 is on the source machine
        # Send src_tensor to rank 1 (destination machine)
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive data from rank 0 (source machine)
        dist.recv(tensor=tensor, src=0)

    # Synchronize destination GPU after receiving
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed = end_time - start_time
    return tensor, elapsed


parser = argparse.ArgumentParser()
parser.add_argument(
    "--address", help="Address of the source machine", type=str, required=True
)
parser.add_argument(
    "--port", help="Port of the source machine", type=str, required=True
)
parser.add_argument("--rank", help="Rank of the machine", type=int, required=True)
parser.add_argument(
    "--device", help="Number of the cuda device", type=int, default=0, required=True
)

args = parser.parse_args()
address = args.address
port = args.port
rank = args.rank
device = args.device

torch.cuda.set_device(device)
# Initialize process group for multi-node communication
dist.init_process_group(
    backend="nccl", init_method=f"tcp://{address}:{port}", rank=rank, world_size=2
)

# Warm Up
send_revc_tensor((3, 3))

# Real Test
for tokens_num in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
    total_elapsed = 0
    for i in range(80):
        tensor, elapsed = send_revc_tensor((tokens_num, 8192))
        total_elapsed += elapsed
        torch.cuda.synchronize()

    # if dist.get_rank() == 0:
    #     print("Send data on source GPU:")
    #     print(tensor)

    # # Print the received data on the destination GPU
    # if dist.get_rank() != 0:  # Assuming process with rank 0 is on the source machine
    #     print("Received data on destination GPU:")
    #     print(tensor)

    print(f"Tokens Num: {tokens_num}, Transfer Time: {total_elapsed}")
