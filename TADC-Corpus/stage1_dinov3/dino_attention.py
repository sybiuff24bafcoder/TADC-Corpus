# -*- coding: utf-8 -*-
"""DINOv3 attention hooks and RoPE utilities used by Stage 1."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

runtime_storage: Dict[int, Dict[str, Any]] = {}


def rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply_manual(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rope_rotate_half(x) * sin)


def apply_rope_to_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    rope: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    sin, cos = rope
    q_dtype = q.dtype
    sin = sin.to(dtype=q_dtype)
    cos = cos.to(dtype=q_dtype)
    n_total = q.shape[-2]
    n_spatial = sin.shape[-2]

    if n_total > n_spatial:
        prefix_len = n_total - n_spatial
        q_prefix = q[:, :, :prefix_len, :]
        q_spatial = q[:, :, prefix_len:, :]
        k_prefix = k[:, :, :prefix_len, :]
        k_spatial = k[:, :, prefix_len:, :]
        q_spatial = rope_apply_manual(q_spatial, sin, cos)
        k_spatial = rope_apply_manual(k_spatial, sin, cos)
        q = torch.cat((q_prefix, q_spatial), dim=-2)
        k = torch.cat((k_prefix, k_spatial), dim=-2)
    else:
        q = rope_apply_manual(q, sin, cos)
        k = rope_apply_manual(k, sin, cos)
    return q, k


def hijack_n_blocks_attention(model: torch.nn.Module, n: int = 4) -> None:
    total_blocks = len(model.blocks)
    start_layer = max(0, total_blocks - n)
    print(f">>> [Hook] Hijacking last {n} layers...")

    def get_new_forward(layer_idx: int):
        original_forward = model.blocks[layer_idx].attn.forward

        def new_forward(x, attn_bias=None, rope=None):
            runtime_storage.setdefault(layer_idx, {})
            runtime_storage[layer_idx]["x"] = x
            runtime_storage[layer_idx]["rope"] = rope
            return original_forward(x, attn_bias=attn_bias, rope=rope)

        return new_forward

    for layer_idx in range(start_layer, total_blocks):
        model.blocks[layer_idx].attn.forward = get_new_forward(layer_idx)


def clear_runtime_storage() -> None:
    runtime_storage.clear()
