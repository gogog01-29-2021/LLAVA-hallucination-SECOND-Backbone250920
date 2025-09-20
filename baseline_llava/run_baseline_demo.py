"""
Demonstration script for the Mini LLaMA baseline.

This script constructs a tiny language model using the MiniLLaMA class
from `mini_llama_numpy.py`, runs a forward pass on a simple batch of
token IDs, prints an example attention matrix, then generates a few
new tokens greedily. The purpose of the demo is to verify that the
model and its layers are wired up correctly.
"""

import numpy as np
from mini_llama_numpy import MiniLLaMA, set_seed, softmax, causal_mask


def main() -> None:
    rng = set_seed(42)
    vocab_size = 20
    model = MiniLLaMA(vocab_size=vocab_size, d_model=16, n_layers=2,
                      n_heads=4, ffn_mult=4.0, rng=rng)
    # A toy batch with a single sequence of four token IDs.
    tokens = np.array([[1, 2, 3, 4]], dtype=np.int32)

    # Hook into the first block to inspect the attention matrix.
    x0 = model.embed[tokens]
    block0 = model.blocks[0]
    # Apply the first RMSNorm.
    x_norm = block0.norm1(x0)
    # Compute q, k, v for the first head only.
    attn = block0.attn
    B, T, d = x_norm.shape
    dh = attn.dh
    q = attn.Wq(x_norm).reshape(B, T, attn.h, dh)
    k = attn.Wk(x_norm).reshape(B, T, attn.h, dh)
    # Apply rotary embeddings.
    q_rot = attn.rope.apply_rotary(q)[:, :, 0, :]
    k_rot = attn.rope.apply_rotary(k)[:, :, 0, :]
    scores = (q_rot @ k_rot.transpose(0, 2, 1)) / np.sqrt(dh)
    scores = scores + causal_mask(T)[None, :, :]
    probs = softmax(scores, axis=-1)
    print("Head 0 attention matrix:\n", np.round(probs[0], 4))

    # Run the full forward pass and print logits of the last token.
    logits = model.forward(tokens)
    print("Last-step logits:\n", np.round(logits[0, -1], 4))

    # Generate three new tokens greedily.
    generated = model.generate(tokens, max_new=3, greedy=True)
    print("Generated token IDs:\n", generated)


if __name__ == "__main__":
    main()
