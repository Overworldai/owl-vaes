import os
import torch.distributed as dist
import datetime as dt

def setup(force=False, timeout=60):
    init_kwargs = dict(timeout=dt.timedelta(seconds=timeout)) if timeout else {}

    if not force:
        try:
            dist.init_process_group(backend="nccl", init_method="env://", **init_kwargs)

            global_rank = int(os.environ.get("RANK", 0))
            local_rank  = int(os.environ.get("LOCAL_RANK", 0))
            world_size  = int(os.environ.get("WORLD_SIZE", 1))

            return global_rank, local_rank, world_size
        except:
            return 0, 0, 1
    else:
        dist.init_process_group(backend="nccl", init_method="env://", **init_kwargs)

        global_rank = int(os.environ.get("RANK", 0))
        local_rank  = int(os.environ.get("LOCAL_RANK", 0))
        world_size  = int(os.environ.get("WORLD_SIZE", 1))

        return global_rank, local_rank, world_size

def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()