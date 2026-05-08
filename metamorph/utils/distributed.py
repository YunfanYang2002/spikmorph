import os

import torch
import torch.distributed as dist

from metamorph.config import cfg


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def init_distributed_mode():
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)

    cfg.RANK = rank
    cfg.LOCAL_RANK = local_rank
    cfg.WORLD_SIZE = world_size
    cfg.DISTRIBUTED = world_size > 1

    if not cfg.DISTRIBUTED:
        return

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
        cfg.DEVICE = "cuda:{}".format(local_rank)
    else:
        backend = "gloo"
        cfg.DEVICE = "cpu"

    dist.init_process_group(
        backend=backend, init_method="env://", rank=rank, world_size=world_size
    )
    synchronize()


def is_distributed():
    return cfg.DISTRIBUTED and dist.is_available() and dist.is_initialized()


def is_main_process():
    return cfg.RANK == 0


def synchronize():
    if is_distributed():
        dist.barrier()


def all_gather_object(data):
    if not is_distributed():
        return [data]
    gathered = [None for _ in range(cfg.WORLD_SIZE)]
    dist.all_gather_object(gathered, data)
    return gathered


def reduce_scalar(value, average=True, op=dist.ReduceOp.SUM):
    if not is_distributed():
        return float(value)

    tensor = torch.tensor(float(value), device=cfg.DEVICE)
    dist.all_reduce(tensor, op=op)
    if average and op == dist.ReduceOp.SUM:
        tensor /= cfg.WORLD_SIZE
    return tensor.item()


def destroy_process_group():
    if is_distributed():
        dist.destroy_process_group()
