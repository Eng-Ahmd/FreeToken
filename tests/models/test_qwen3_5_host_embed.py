"""Host-pinned embeddings: the lookup table stays in host RAM and is gathered
zero-copy (UVA), saving ~2.5GB of VRAM on dense checkpoints.

Pinning itself is covered with a stubbed allocator (deterministic without the built
extension) plus a real-alloc case that runs where the extension exists; the gather
kernel has its own numeric test in tests/kernels/test_embed_gather.py.
"""

from __future__ import annotations

import pytest
import torch

_EMBED = "model.embed_tokens.weight"
_NORM = "model.layers.0.input_layernorm.weight"


def _model_state() -> dict[str, torch.Tensor]:
    return {
        _EMBED: torch.empty(32, 16, dtype=torch.bfloat16),
        _NORM: torch.empty(16, dtype=torch.bfloat16),
    }


def test_materialize_pins_embed_when_alloc_works(monkeypatch):
    engine = pytest.importorskip("freetoken.engine.engine")
    calls: list = []

    def fake_alloc(*shape: int, dtype: torch.dtype) -> torch.Tensor:
        calls.append((shape, dtype))
        return torch.empty(*shape, dtype=dtype)

    monkeypatch.setattr("freetoken.kernel.pinned.alloc_pinned_tensor", fake_alloc)
    table = torch.randn(32, 16, dtype=torch.bfloat16)
    state_dict = engine._materialize_loaded_weight_state_dict(
        _model_state(), [(_EMBED, table)], device=torch.device("cpu")
    )
    assert calls == [((32, 16), torch.bfloat16)]
    assert state_dict[_EMBED].device.type == "cpu"
    assert torch.equal(state_dict[_EMBED], table)


def test_materialize_falls_back_to_device_when_alloc_fails(monkeypatch):
    engine = pytest.importorskip("freetoken.engine.engine")

    def _boom(*shape: int, dtype: torch.dtype) -> torch.Tensor:
        raise RuntimeError("no pin quota")

    monkeypatch.setattr("freetoken.kernel.pinned.alloc_pinned_tensor", _boom)
    table = torch.randn(32, 16, dtype=torch.bfloat16)
    norm = torch.randn(16, dtype=torch.bfloat16)
    state_dict = engine._materialize_loaded_weight_state_dict(
        _model_state(), [(_EMBED, table), (_NORM, norm)], device=torch.device("cpu")
    )
    assert state_dict[_EMBED].device.type == "cpu"
    assert torch.equal(state_dict[_EMBED], table)
    assert torch.equal(state_dict[_NORM], norm)


def _pinned_extension_available() -> bool:
    try:
        import freetoken.kernel._pinned_tensor  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _pinned_extension_available(),
    reason="needs CUDA + the built pinned-tensor extension",
)
def test_materialize_really_pins_off_gpu():
    engine = pytest.importorskip("freetoken.engine.engine")
    table = torch.randn(32, 16, dtype=torch.bfloat16, device="cuda")
    state_dict = engine._materialize_loaded_weight_state_dict(
        _model_state(), [(_EMBED, table)], device=torch.device("cuda")
    )
    out = state_dict[_EMBED]
    assert out.device.type == "cpu"
    assert out.is_pinned()
    assert torch.equal(out, table.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_materialize_still_moves_other_weights_to_gpu():
    engine = pytest.importorskip("freetoken.engine.engine")
    norm = torch.randn(16, dtype=torch.bfloat16)
    state_dict = engine._materialize_loaded_weight_state_dict(
        _model_state(), [(_NORM, norm)], device=torch.device("cuda")
    )
    assert state_dict[_NORM].device.type == "cuda"
    assert torch.equal(state_dict[_NORM].cpu(), norm)
