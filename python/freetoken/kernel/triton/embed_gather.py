"""Host-table embedding gather (zero-copy UVA).

The TVM ``indexing`` kernel requires weights and indices on the same device, so a
host-pinned embedding table (``model.embed_tokens.weight`` kept off the GPU to save
~2.5GB VRAM) needs its own gather: one program per token row, reading the row
straight from pinned host RAM. Only looked-up rows cross PCIe (~10KB/token);
measured 0.040ms vs 0.030ms VRAM-resident per 8-row gather on sm_120 (bit-exact).

Fixed shapes, fixed pointers, no host sync: CUDA-graph safe.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_kernel(
    w_ptr, idx_ptr, out_ptr, H,
    stride_w, stride_o,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    row = tl.load(idx_ptr + pid).to(tl.int64)
    offs = tl.arange(0, BLOCK_H)
    v = tl.load(w_ptr + row * stride_w + offs, mask=offs < H)
    tl.store(out_ptr + pid * stride_o + offs, v, mask=offs < H)


def embed_gather_host(weights: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather rows of a CPU (pinned-host) bf16 table by GPU ``indices``.

    ``weights`` ``[vocab, H]`` CPU, ``indices`` ``[N]`` int64 on CUDA, returns
    ``[N, H]`` bf16 on CUDA.
    """
    n, h = indices.shape[0], weights.shape[1]
    out = torch.empty((n, h), dtype=weights.dtype, device=indices.device)
    _gather_kernel[(n,)](
        weights, indices, out, h,
        weights.stride(0), out.stride(0),
        BLOCK_H=triton.next_power_of_2(h),
    )
    return out


__all__ = ["embed_gather_host"]
