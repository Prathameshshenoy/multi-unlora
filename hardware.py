"""
hardware.py

CUDA device management and VRAM introspection utilities.

Centralises all direct interaction with the CUDA runtime so the rest of the
codebase never calls torch.cuda APIs directly. Provides memory flushing at
training boundaries, VRAM measurement for the paper's infrastructure metrics,
and a startup sanity check to confirm the expected GPU was allocated.
"""

import gc
import torch


def flush_vram():
    """Releases the PyTorch CUDA memory pool and runs Python garbage collection.

    Intended to be called at explicit boundaries — most importantly between
    training the two BadExpert adapters — to prevent residual allocations from
    one training run from fragmenting the memory arena of the next.

    On Modal each function runs in an isolated container, so mid-loop flushing
    is unnecessary. ipc_collect is included for correctness in multi-process
    contexts, though it is a no-op in single-process Modal containers.

    Side effects:
        Frees all unreferenced Python objects, empties the CUDA block cache,
        and releases any inter-process CUDA handle references.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def get_vram_usage_mb() -> float:
    """Returns the amount of CUDA memory currently allocated by PyTorch, in MB.

    Used in runners.py to compute the per-example VRAM delta attributable to
    the MultiUnLoRALogitsProcessor's adapter forward passes.

    Returns:
        float: Allocated CUDA memory in megabytes.
    """
    return torch.cuda.memory_allocated() / (1024**2)


def get_vram_reserved_mb() -> float:
    """Returns the amount of CUDA memory reserved by the PyTorch allocator, in MB.

    Reserved memory is always greater than or equal to allocated memory.
    PyTorch retains freed blocks in its internal cache to avoid the overhead
    of round-tripping back to the CUDA driver for every allocation.

    Returns:
        float: Reserved CUDA memory in megabytes.
    """
    return torch.cuda.memory_reserved() / (1024**2)


def log_device():
    """Logs the active GPU name and total memory to stdout.

    Called at the start of every GPU Modal function as a sanity check that
    the requested hardware tier was actually allocated. Prints a warning and
    returns without error if no CUDA device is available, allowing CPU-only
    execution paths to proceed.

    Side effects:
        Prints one line to stdout describing the active GPU, or a warning
        if no CUDA device is found.
    """
    if not torch.cuda.is_available():
        print("WARNING: No CUDA device found. Running on CPU.")
        return
    name = torch.cuda.get_device_name(0)
    total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    print(f"Active GPU: {name} ({total_mb:.0f} MB)")
