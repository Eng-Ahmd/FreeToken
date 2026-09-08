"""qwen3_5_moe plain-ModelOpt-NVFP4 dense support (e.g. Qwen3.8-27B-NVFP4-RTX5090).

Config parses off trimmed copies of the real checkpoint shapes via RawConfigShim (the
object cached_load_hf_config falls back to); the fusion test drives _attn_nvfp4_emit
with synthetic ModelOpt-layout tensors, so no checkpoint or GPU is needed except for
the numeric check.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.models.qwen3_5_moe.config import (
    _modelopt_plain_nvfp4,
    parse_config,
)
from freetoken.utils.hf import RawConfigShim

_HIDDEN = 5120
_INTER = 17408
_VOCAB = 248320
_LAYERS = 64


def _text_config(moe: bool = False) -> dict:
    cfg: dict = {
        "hidden_size": _HIDDEN,
        "intermediate_size": _INTER,
        "num_hidden_layers": _LAYERS,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "vocab_size": _VOCAB,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 262144,
        "tie_word_embeddings": False,
        "head_dim": 256,
        "partial_rotary_factor": 0.25,
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "full_attention_interval": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
    }
    if moe:
        cfg.update({"num_experts": 256, "num_experts_per_tok": 8})
    return cfg


def _hf_config(text: dict, quantization_config: dict | None = None) -> RawConfigShim:
    data: dict = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": text,
    }
    if quantization_config is not None:
        data["quantization_config"] = quantization_config
    return RawConfigShim(data)


_PLAIN_NVFP4_QUANT = {
    # gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090: single NVFP4 algo, exclude-list.
    "quant_algo": "NVFP4",
    "kv_cache_quant_algo": "FP8",
    "group_size": 16,
    "exclude_modules": [
        "model.language_model.embed_tokens",
        "model.language_model.layers.0.linear_attn.conv1d",
        "model.language_model.layers.0.linear_attn.in_proj_a",
        "model.language_model.layers.0.linear_attn.in_proj_b",
        "model.visual*",
    ],
}

_MIXED_QUANT = {
    # Ornith/Qwen3.6-MoE shape: top-level MIXED_PRECISION + per-layer map.
    "quant_algo": "MIXED_PRECISION",
    "quant_method": "modelopt",
    "quantized_layers": {
        "model.language_model.layers.3.self_attn.q_proj": {"quant_algo": "FP8"},
        "model.language_model.layers.3.mlp.experts": {
            "quant_algo": "W4A16_NVFP4",
            "group_size": 16,
        },
    },
}


def test_plain_nvfp4_dense_keeps_attention_and_head_native():
    cfg = parse_config(_hf_config(_text_config(), _PLAIN_NVFP4_QUANT))
    assert not cfg.moe_enabled
    assert cfg.expert_quant == "nvfp4"
    assert cfg.attn_quant == "nvfp4"
    assert cfg.dense_quant == "nvfp4"
    assert cfg.lm_head_quant == "nvfp4"


def test_plain_nvfp4_lm_head_exclusion_is_honored():
    quant = dict(_PLAIN_NVFP4_QUANT)
    quant["exclude_modules"] = [*quant["exclude_modules"], "lm_head"]
    cfg = parse_config(_hf_config(_text_config(), quant))
    assert (cfg.attn_quant, cfg.dense_quant) == ("nvfp4", "nvfp4")
    assert cfg.lm_head_quant == "none"


def test_plain_detector_rejects_mixed_compressed_and_bf16():
    assert not _modelopt_plain_nvfp4(_hf_config(_text_config(), _MIXED_QUANT))
    assert not _modelopt_plain_nvfp4(_hf_config(_text_config(), None))
    ct = {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                    "strategy": "tensor_group",
                },
            }
        },
    }
    assert not _modelopt_plain_nvfp4(_hf_config(_text_config(), ct))


def test_moe_checkpoint_never_takes_the_plain_dense_path():
    # Pure-NVFP4-shaped MoE (top-level NVFP4 algo, no per-layer map): the moe_enabled
    # gate keeps attention on its existing path even though the detector matches.
    quant = {"quant_algo": "NVFP4", "group_size": 16, "exclude_modules": []}
    cfg = parse_config(_hf_config(_text_config(moe=True), quant))
    assert cfg.moe_enabled
    assert cfg.attn_quant == "none"
    assert cfg.dense_quant == "nvfp4"
    assert cfg.lm_head_quant == "none"


def test_mixed_moe_parse_is_unchanged():
    cfg = parse_config(_hf_config(_text_config(moe=True), _MIXED_QUANT))
    assert cfg.moe_enabled
    assert (cfg.expert_quant, cfg.attn_quant) == ("nvfp4", "fp8_pertensor")
    assert cfg.lm_head_quant == "none"


class _FakeWeights:
    """Minimal ``f`` for _attn_nvfp4_emit: get_tensor over a dict."""

    def __init__(self, tensors: dict[str, torch.Tensor]):
        self._tensors = tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


def _nvfp4_triple(rows: int, cols: int, seed: int, global_scale: float = 1.5):
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=g)
    # Finite fp8-e4m3 block scales only (0x7F/0xFF are NaN; checkpoints never store them).
    scale = torch.randint(0, 0x7E, (rows, cols // 16), dtype=torch.uint8, generator=g).view(
        torch.float8_e4m3fn
    )
    glob = torch.tensor(global_scale, dtype=torch.float16)  # per-tensor scalar
    return packed, scale, glob


def _emit_case():
    from freetoken.models.qwen3_5_moe.weight import _attn_nvfp4_emit

    # q/k/v with distinct per-part globals: the fused triple must preserve each part.
    parts = {
        ".self_attn.q_proj": _nvfp4_triple(12288, 5120, 1, 1.5),
        ".self_attn.k_proj": _nvfp4_triple(1024, 5120, 2, 2.0),
        ".self_attn.v_proj": _nvfp4_triple(1024, 5120, 3, 0.75),
    }
    tensors: dict[str, torch.Tensor] = {}
    for suffix, (w, s, g) in parts.items():
        tensors[f"model.layers.0{suffix}.weight"] = w
        tensors[f"model.layers.0{suffix}.weight_scale"] = s
        tensors[f"model.layers.0{suffix}.weight_scale_2"] = g
    f = _FakeWeights(tensors)
    buf: dict = {}
    out = None
    for suffix in (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"):
        out = _attn_nvfp4_emit(
            f, f"model.layers.0{suffix}", f"model.layers.0{suffix}", buf=buf
        )
    return parts, buf, out


def test_attn_qkv_fusion_is_exact_and_buffered():
    parts, buf, out = _emit_case()
    assert buf == {}
    assert out is not None and len(out) == 3
    by_key = dict(out)
    w = by_key["model.layers.0.self_attn.qkv_proj.weight"]
    s = by_key["model.layers.0.self_attn.qkv_proj.weight_scale"]
    g = by_key["model.layers.0.self_attn.qkv_proj.weight_global"]
    assert w.shape == (12288 + 1024 + 1024, 5120 // 2)
    assert s.shape == (12288 + 1024 + 1024, 5120 // 16)
    assert g.shape == (14336,)
    off = 0
    for suffix in (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"):
        pw, ps, pg = parts[suffix]
        n = pw.shape[0]
        assert torch.equal(w[off : off + n], pw)
        assert torch.equal(s[off : off + n].view(torch.uint8), ps.view(torch.uint8))
        assert torch.equal(g[off : off + n], pg.reshape(1).to(torch.float16).expand(n))
        off += n


def test_ba_fusion_order_matches_model_split():
    from freetoken.models.qwen3_5_moe.weight import _NVFP4_BF16_FUSE, _try_fuse

    buf: dict = {}
    b = torch.randn(48, 5120)
    a = torch.randn(48, 5120)
    assert _try_fuse("model.layers.0.linear_attn.in_proj_b.weight", b, buf, _NVFP4_BF16_FUSE) == ()
    key, merged = _try_fuse(
        "model.layers.0.linear_attn.in_proj_a.weight", a, buf, _NVFP4_BF16_FUSE
    )
    assert key == "model.layers.0.linear_attn.in_proj_ba.weight"
    assert torch.equal(merged, torch.cat([b, a], dim=0))
    assert buf == {}


def test_attn_emit_rejects_non_attention_and_buffers_partials():
    from freetoken.models.qwen3_5_moe.weight import (
        _NOT_ATTN_NVFP4,
        _attn_nvfp4_emit,
    )

    w, s, g = _nvfp4_triple(512, 5120, 7)
    f = _FakeWeights(
        {
            "model.layers.0.mlp.gate_proj.weight": w,
            "model.layers.0.mlp.gate_proj.weight_scale": s,
            "model.layers.0.mlp.gate_proj.weight_scale_2": g.reshape(1),
        }
    )
    assert (
        _attn_nvfp4_emit(
            f, "model.layers.0.mlp.gate_proj", "model.layers.0.mlp.gate_proj", buf={}
        )
        is _NOT_ATTN_NVFP4
    )
    wq, sq, gq = _nvfp4_triple(12288, 5120, 9)
    fq = _FakeWeights(
        {
            "model.layers.0.self_attn.q_proj.weight": wq,
            "model.layers.0.self_attn.q_proj.weight_scale": sq,
            "model.layers.0.self_attn.q_proj.weight_scale_2": gq,
        }
    )
    buf: dict = {}
    assert (
        _attn_nvfp4_emit(
            fq, "model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.q_proj", buf=buf
        )
        == []
    )
    assert len(buf) == 1  # partial fusion stays buffered, nothing emitted


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA dequant kernel")
def test_fused_native_dequant_matches_per_part_dequant():
    from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4

    parts, _, out = _emit_case()
    by_key = dict(out)
    dev = torch.device("cuda")
    w = by_key["model.layers.0.self_attn.qkv_proj.weight"].to(dev)
    s = by_key["model.layers.0.self_attn.qkv_proj.weight_scale"].to(dev)
    g = by_key["model.layers.0.self_attn.qkv_proj.weight_global"].to(dev)
    slots = torch.zeros(1, dtype=torch.int32, device=dev)
    fused = dequant_nvfp4(
        w.unsqueeze(0).contiguous(),
        s.unsqueeze(0).contiguous(),
        g.unsqueeze(0),
        slots,
        dtype=torch.bfloat16,
    )[0]
    off = 0
    for suffix in (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"):
        pw, ps, pg = parts[suffix]
        n = pw.shape[0]
        part = dequant_nvfp4(
            pw.unsqueeze(0).to(dev).contiguous(),
            ps.unsqueeze(0).to(dev).contiguous(),
            pg.reshape(1).to(torch.float16).expand(n).unsqueeze(0).to(dev),
            slots,
            dtype=torch.bfloat16,
        )[0]
        assert torch.equal(fused[off : off + n], part)
        off += n
