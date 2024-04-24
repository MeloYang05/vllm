import torch
import time
import os
import torch.distributed as dist


def setup(rank, world_size):
    # Set CUDA device so GPUs allocate tensor on the right device
    torch.cuda.set_device(rank)

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def tensor_transfer(src_gpu, dest_gpu, tensor, recv_tensor):
    # Assuming tensor is on the src_gpu, transfer it to dest_gpu
    if dist.get_rank() == src_gpu:
        # Send tensor to dest_gpu
        dist.send(tensor, dest_gpu)
    elif dist.get_rank() == dest_gpu:
        # Tensor placeholder for received tensor
        dist.recv(recv_tensor, src_gpu)
        return recv_tensor
    return None


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup(rank, world_size)

    # Example tensor
    tensor_warmup = torch.randn(
        10, device=rank, dtype=torch.float16
    )  # Tensor Transfer for warming up

    # Warm Up
    if dist.get_rank() == 1:
        tmp_tensor = torch.zeros_like(tensor_warmup)
    else:
        tmp_tensor = None
    transferred_tensor = tensor_transfer(
        src_gpu=0, dest_gpu=1, tensor=tensor_warmup, recv_tensor=tmp_tensor
    )

    # Transfer Layer By Layer
    for tokens_num in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        tensors = []
        tmp_tensors = []
        for i in range(80):
            tensor = torch.randn(
                    tokens_num * (8192), device=rank, dtype=torch.float16
                )
            tensors.append(tensor)
            if dist.get_rank() == 1:
                tmp_tensor = torch.zeros_like(tensor)
            else:
                tmp_tensor = None
            tmp_tensors.append(tmp_tensor)
        start_time = time.time()
        for i in range(80):   
            transferred_tensor = tensor_transfer(
                src_gpu=0, dest_gpu=1, tensor=tensors[i], recv_tensor=tmp_tensors[i]
            )
            torch.cuda.synchronize()
        end_time = time.time()
        elapsed = end_time - start_time

        if transferred_tensor is not None:
            # print(f"Received tensor on GPU {dist.get_rank()}: {transferred_tensor}")
            # print(f"Received tensor shape on GPU {dist.get_rank()}: {transferred_tensor.shape}")
            print(f"Tokens Num: {tokens_num}, Transfer Time: {elapsed}")

    # Complete Transfer
    # for tokens_num in [8192]:
    #     tensor = torch.randn(
    #         tokens_num * (80 * 8192), device=rank, dtype=torch.float16
    #     )  # Tensor is allocated on the current GPU
    #     if dist.get_rank() == 1:
    #         tmp_tensor = torch.zeros_like(tensor)
    #     else:
    #         tmp_tensor = None
    #     start_time = time.time()
    #     transferred_tensor = tensor_transfer(
    #         src_gpu=0, dest_gpu=1, tensor=tensor, recv_tensor=tmp_tensor
    #     )
    #     torch.cuda.synchronize()
    #     end_time = time.time()
    #     elapsed = end_time - start_time

    #     if transferred_tensor is not None:
    #         # print(f"Received tensor on GPU {dist.get_rank()}: {transferred_tensor}")
    #         # print(f"Received tensor shape on GPU {dist.get_rank()}: {transferred_tensor.shape}")
    #         print(f"Tokens Num: {tokens_num}, Transfer Time: {elapsed}")

    cleanup()


if __name__ == "__main__":
    main()
