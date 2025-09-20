"""
Demonstration script for SECOND hooks.

This demo shows how to use the helper functions provided in
``second_hooks.py`` without requiring a full LLaVA implementation. It
constructs dummy tensors for positional embeddings, attention weights
and logits and applies interpolation, top‑k masking, and contrastive
mixing. The intent is to illustrate the API and behaviour of the
helpers rather than to perform meaningful vision–language inference.
"""

import torch
import numpy as np

from second_hooks import (
    interpolate_pos_encoding,
    topk_mask_from_attn,
    contrastive_mix,
)


def demo_interpolate_pos_encoding() -> None:
    # Create a dummy positional embedding for a 4×4 grid with dim=8 plus CLS.
    S = 4
    D = 8
    pos = torch.arange(1 + S * S * D, dtype=torch.float32).reshape(1, 1 + S * S, D)
    # Interpolate to a 2×2 grid.
    out = interpolate_pos_encoding(pos, 2, 2)
    print("Interpolated pos embed shape:", out.shape)


def demo_topk_mask_from_attn() -> None:
    # Create fake attention weights: layers=2, heads=3, T_text=1, N_visual=6.
    attn = torch.rand(2, 3, 1, 6)
    mask = topk_mask_from_attn(attn, topk=0.5)  # keep top 50%
    print("Top‑k mask:", mask)


def demo_contrastive_mix() -> None:
    # Create dummy logits for three stages (vocab size=5).
    logits_list = [torch.rand(5) for _ in range(3)]
    alphas = [0.1, 0.2, 0.7]
    mixed = contrastive_mix(logits_list, alphas)
    print("Contrastively mixed logits:", mixed)


if __name__ == "__main__":
    demo_interpolate_pos_encoding()
    demo_topk_mask_from_attn()
    demo_contrastive_mix()
