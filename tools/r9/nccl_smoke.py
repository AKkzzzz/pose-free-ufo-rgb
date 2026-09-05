#!/usr/bin/env python3
"""Minimal eight-rank NCCL all-reduce smoke."""

import os

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    value = torch.tensor(float(rank + 1), device="cuda")
    dist.all_reduce(value)
    expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"NCCL all-reduce mismatch: {value.item()} != {expected}")
    dist.barrier()
    if rank == 0:
        print(f"NCCL_SMOKE=PASS world_size={dist.get_world_size()} sum={value.item()}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
