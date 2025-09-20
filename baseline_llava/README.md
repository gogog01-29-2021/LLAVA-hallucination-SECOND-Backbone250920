# Mini LLaVA Baseline

This directory contains a small, self‑contained implementation of a
decoder‑only Transformer written in pure NumPy. It is **not** the
full‑sized LLaVA model, but a toy model that follows similar design
principles: RMS normalization, rotary positional embeddings, multi‑head
self‑attention with a causal mask, and a SwiGLU feed‑forward network.

The core implementation lives in `mini_llama_numpy.py`. You can use
`run_baseline_demo.py` to instantiate the model, run a forward pass
over a toy batch, and generate a few tokens to verify that the model
works end‑to‑end. Because this is a small model with randomly
initialized weights, the outputs are not meaningful text. The goal is
to provide a working baseline for experimenting with Transformer
architectures.

## Files

| File | Description |
|-----|-------------|
| `mini_llama_numpy.py` | NumPy implementation of a tiny LLaMA‑style decoder. |
| `run_baseline_demo.py` | Example script to run a forward pass and generation demo. |

## Usage

Install NumPy if it is not already available (most Python
environments include it by default). Then run the demo script:

```bash
python run_baseline_demo.py
```

The script will print the attention matrix for the first attention
head, the logits for the last token, and a short sequence of
generated token IDs. Feel free to modify the model hyperparameters in
`run_baseline_demo.py` to explore how depth, width, and head count
affect performance.
