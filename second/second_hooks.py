"""
SECOND hooks and utilities.

This module contains helper functions that implement the key
components of the SECOND (Selective and Contrastive Decoding) method
for mitigating perceptual hallucinations in vision–language models. The
functions defined here are model‑agnostic and operate on logits,
attention weights, and positional embeddings. They can be composed
with any decoder‑only language model to experiment with multi‑stage
generation, selective integration based on attention, and contrastive
logit mixing across stages.
"""

import torch
import torch.nn.functional as F


def interpolate_pos_encoding(pos_embed: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Bilinearly interpolate a 2‑D positional embedding to a new grid size.

    Args:
        pos_embed: Positional embedding tensor of shape [1, 1 + S*S, D],
            where the first token is the CLS token and the remainder is a
            flattened 2‑D grid of shape (S,S,D).
        h: Height of the target grid.
        w: Width of the target grid.

    Returns:
        A tensor of shape [1, 1 + h*w, D] with the CLS token unchanged
        and the grid reshaped and interpolated to (h,w).
    """
    cls = pos_embed[:, :1]
    grid = pos_embed[:, 1:]
    s = int(grid.shape[1] ** 0.5)
    grid = grid.reshape(1, s, s, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(h, w), mode="bilinear", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, h * w, -1)
    return torch.cat([cls, grid], dim=1)


def topk_mask_from_attn(attn: torch.Tensor, topk: float | int) -> torch.BoolTensor:
    """Build a mask for the top‑k visual tokens based on attention weights.

    Args:
        attn: Attention weights of shape [layers, heads, T_text, N_visual].
        topk: Either an integer specifying the exact number of tokens to keep
            or a float in (0,1) specifying the proportion of tokens to keep.

    Returns:
        A boolean mask of shape [N_visual] where True indicates the token
        should be kept for the next stage.
    """
    w = attn.mean(dim=(0, 1, 2))
    if isinstance(topk, int):
        k = topk
    else:
        k = max(1, int(w.numel() * topk))
    idx = torch.topk(w, k).indices
    mask = torch.zeros_like(w, dtype=torch.bool)
    mask[idx] = True
    return mask


def contrastive_mix(logits_list: list[torch.Tensor], alphas: list[float]) -> torch.Tensor:
    """Mix logits from multiple stages using contrastive alpha weights.

    Args:
        logits_list: A list of tensors of shape [V] (one per stage).
        alphas: A list of floats of the same length as ``logits_list``.

    Returns:
        A tensor of shape [V] representing the weighted sum of logits.
    """
    assert len(logits_list) == len(alphas)
    L = torch.stack(logits_list, dim=0)
    A = torch.tensor(alphas, device=L.device, dtype=L.dtype)
    A = A / (A.sum() + 1e-8)
    return (A[:, None] * L).sum(dim=0)


def run_second(model_llava, image, prompt, stages, grid=None,
               pos_interp: bool = True, attn_topk_ratio: float | None = None,
               contrastive_alphas: list[float] | None = None) -> torch.Tensor:
    """Run SECOND on top of a LLaVA model.

    This function is a high‑level driver that illustrates how the hooks
    defined above can be composed. It assumes that ``model_llava``
    exposes ``build_patches``, ``encode_vision`` and ``decode_step``
    methods; these abstractions correspond to the AnyRes tiling,
    vision tower forward pass, and language model decoding. The
    function iterates over the given ``stages``, optionally performs
    positional encoding interpolation, filters visual tokens based on
    attention, and mixes logits with contrastive weights.

    Note: This code is provided as a blueprint and will require
    adaptation to the specific LLaVA implementation you are using.
    """
    logits_per_stage = []
    visual_carry = None
    for s in stages:
        patches = model_llava.build_patches(image, grid=grid, stage=s)
        vtoks, pos = model_llava.encode_vision(patches)
        if pos_interp:
            H, W = model_llava.grid_for_stage(s)
            pos = interpolate_pos_encoding(pos, H, W)
        logits, attn = model_llava.decode_step(prompt, vtoks, pos,
                                               visual_carry, need_attn=True)
        logits_per_stage.append(logits)
        if attn_topk_ratio is not None:
            mask = topk_mask_from_attn(attn, attn_topk_ratio)
            visual_carry = {"mask": mask}
    if contrastive_alphas is not None:
        final_logits = contrastive_mix(logits_per_stage, contrastive_alphas)
    else:
        final_logits = logits_per_stage[-1]
    return final_logits
