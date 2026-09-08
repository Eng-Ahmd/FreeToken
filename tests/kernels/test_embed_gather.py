"""Host-table embedding gather: bit-exact vs a VRAM-resident reference gather."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

_H = 5120


def _reference_gather(weights: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return weights.to(indices.device)[indices]


@pytest.mark.parametrize("rows", [1, 8, 64])
def test_host_gather_matches_device_gather(rows: int):
    from freetoken.kernel.triton.embed_gather import embed_gather_host

    dev = torch.device("cuda")
    table = torch.randn(4096, _H, dtype=torch.bfloat16, device=dev)
    host = torch.empty(4096, _H, dtype=torch.bfloat16, pin_memory=True)
    host.copy_(table)
    idx = torch.randint(0, 4096, (rows,), dtype=torch.int64, device=dev)
    got = embed_gather_host(host, idx)
    assert got.shape == (rows, _H) and got.dtype == torch.bfloat16
    assert got.device.type == "cuda"
    assert torch.equal(got, _reference_gather(table, idx))
