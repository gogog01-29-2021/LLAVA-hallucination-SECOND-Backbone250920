"""
Mini LLaMA implementation in pure NumPy.

This module implements a tiny decoder-only Transformer similar in spirit
to Meta's LLaMA family. It features the key architectural
components—RMSNorm, rotary position embeddings (RoPE), multi‑head
attention with a causal mask, and a SwiGLU feed‑forward network. The
implementation is intended for educational purposes and mirrors the
mathematical description of a Transformer.

The code is self contained and has no external dependencies beyond
NumPy. It is small enough to run quickly on CPUs but structured to
match a real LLM's forward pass.

See ``run_baseline_demo.py`` for an example of how to instantiate and
exercise this model.
"""

import math
import numpy as np


def set_seed(seed: int = 0) -> np.random.Generator:
    """Return a deterministic NumPy random generator."""
    return np.random.default_rng(seed)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """A numerically stable softmax implementation."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def causal_mask(T: int) -> np.ndarray:
    """Return a [T, T] mask with -1e9 above the diagonal and 0 elsewhere."""
    m = np.triu(np.ones((T, T), dtype=bool), k=1)
    mask = np.zeros((T, T), dtype=np.float32)
    mask[m] = -1e9
    return mask


class RMSNorm:
    """Root mean square normalization layer."""

    def __init__(self, d: int, eps: float = 1e-5) -> None:
        self.g = np.ones((d,), dtype=np.float32)
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.g


class RoPE:
    """Rotary positional embeddings for one head dimension."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.base = base

    def get_sin_cos(self, T: int) -> tuple[np.ndarray, np.ndarray]:
        d = self.head_dim
        freqs = (1.0 / (self.base ** (np.arange(0, d, 2) / d))).astype(np.float32)
        pos = np.arange(T, dtype=np.float32)[:, None]
        angles = pos * freqs[None, :]
        return np.sin(angles), np.cos(angles)

    def apply_rotary(self, x: np.ndarray) -> np.ndarray:
        B, T, H, Dh = x.shape
        sin, cos = self.get_sin_cos(T)
        sin = sin[None, :, None, :]
        cos = cos[None, :, None, :]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        x_rot_even = x1 * cos - x2 * sin
        x_rot_odd = x1 * sin + x2 * cos
        out = np.empty_like(x)
        out[..., 0::2] = x_rot_even
        out[..., 1::2] = x_rot_odd
        return out


class Linear:
    """A simple linear layer with optional bias."""

    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator,
                 bias: bool = False, std: float | None = None) -> None:
        scale = std if std is not None else (1.0 / math.sqrt(in_dim))
        self.W = (rng.standard_normal((in_dim, out_dim)).astype(np.float32) * scale)
        self.b = np.zeros((out_dim,), dtype=np.float32) if bias else None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = x @ self.W
        if self.b is not None:
            y = y + self.b
        return y


class MultiHeadAttention:
    """Multi‑head self‑attention with rotary embeddings and causal masking."""

    def __init__(self, d_model: int, n_heads: int, rng: np.random.Generator) -> None:
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d = d_model
        self.h = n_heads
        self.dh = d_model // n_heads
        self.Wq = Linear(d_model, d_model, rng, bias=False)
        self.Wk = Linear(d_model, d_model, rng, bias=False)
        self.Wv = Linear(d_model, d_model, rng, bias=False)
        self.Wo = Linear(d_model, d_model, rng, bias=False)
        self.rope = RoPE(self.dh)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        B, T, _ = x.shape
        q = self.Wq(x).reshape(B, T, self.h, self.dh)
        k = self.Wk(x).reshape(B, T, self.h, self.dh)
        v = self.Wv(x).reshape(B, T, self.h, self.dh)
        q = self.rope.apply_rotary(q)
        k = self.rope.apply_rotary(k)
        att = np.einsum('bthd,bshd->bhts', q, k) / math.sqrt(self.dh)
        m = causal_mask(T)
        att = att + m[None, None, :, :]
        P = softmax(att, axis=-1)
        y = np.einsum('bhts,bshd->bthd', P, v)
        y = y.reshape(B, T, self.d)
        return self.Wo(y)


class SwiGLU:
    """SwiGLU feed‑forward network used in LLaMA."""

    def __init__(self, d_model: int, mult: float, rng: np.random.Generator) -> None:
        inner = int(mult * d_model)
        self.W1 = Linear(d_model, inner, rng, bias=False)
        self.W2 = Linear(d_model, inner, rng, bias=False)
        self.W3 = Linear(inner, d_model, rng, bias=False)

    @staticmethod
    def silu(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-x))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        u = self.W1(x)
        v = self.W2(x)
        return self.W3(self.silu(v) * u)


class TransformerBlock:
    """A decoder block with attention and feed‑forward sublayers."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: float,
                 rng: np.random.Generator) -> None:
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, rng)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_mult, rng)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MiniLLaMA:
    """A miniature language model loosely inspired by LLaMA."""

    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 2,
                 n_heads: int = 4, ffn_mult: float = 4.0,
                 rng: np.random.Generator | None = None,
                 tie_weights: bool = True) -> None:
        self.rng = rng or set_seed(0)
        self.vocab = vocab_size
        self.d = d_model
        self.embed = (self.rng.standard_normal((vocab_size, d_model)).astype(np.float32)
                      * (1.0 / math.sqrt(d_model)))
        self.blocks = [TransformerBlock(d_model, n_heads, ffn_mult, self.rng)
                       for _ in range(n_layers)]
        self.W_out = self.embed if tie_weights else (
            self.rng.standard_normal((vocab_size, d_model)).astype(np.float32)
            * (1.0 / math.sqrt(d_model))
        )

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        B, T = tokens.shape
        x = self.embed[tokens]
        for blk in self.blocks:
            x = blk(x)
        logits = np.einsum('btd,vd->btv', x, self.W_out)
        return logits

    def generate(self, tokens: np.ndarray, max_new: int = 10,
                 greedy: bool = True, temperature: float = 1.0) -> np.ndarray:
        for _ in range(max_new):
            logits = self.forward(tokens)
            last = logits[:, -1, :] / max(1e-6, temperature)
            if greedy:
                next_tok = np.argmax(last, axis=-1, keepdims=True)
            else:
                probs = softmax(last, axis=-1)
                B = probs.shape[0]
                next_tok = np.zeros((B, 1), dtype=np.int32)
                for b in range(B):
                    next_tok[b, 0] = np.random.choice(self.vocab, p=probs[b])
            tokens = np.concatenate([tokens, next_tok], axis=1)
        return tokens


class VisualProjector:
    """Project visual features to the model dimension and prepend them."""

    def __init__(self, d_vision: int, d_model: int, rng: np.random.Generator) -> None:
        self.proj = Linear(d_vision, d_model, rng, bias=False)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.proj(z)


def prepend_visual_tokens(text_tokens: np.ndarray, visual_feats: np.ndarray,
                          projector: VisualProjector, model: MiniLLaMA) -> np.ndarray:
    B, T = text_tokens.shape
    zt = projector(visual_feats)
    xt = model.embed[text_tokens]
    return np.concatenate([zt, xt], axis=1)
