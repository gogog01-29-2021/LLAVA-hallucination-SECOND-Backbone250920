# Unified LLava/Grok/SECOND Example

This project brings together three independent pieces of code in a
single directory structure to make it easy to explore different
large‑language‑model ideas side by side. The goal is not to provide
production‑ready models but rather a sandbox where you can run a
simple Transformer baseline, play with the SECOND hooks for selective
and contrastive decoding, and examine the skeleton of a Grok‑1 driver
script.

## Directory layout

| Path | Description |
|------|-------------|
| `baseline_llava/` | A tiny, fully self‑contained Transformer implemented in pure NumPy. Use this as a baseline language model. |
| `second/` | A PyTorch module implementing the core functions from the SECOND paper: positional embedding interpolation, top‑k attention masking and contrastive logit mixing. Includes a demo script. |
| `grok/` | A snapshot of the `run.py` script from the Grok‑1 example repository. It shows how to set up and run the Grok model, though the full dependencies are not included here. |

## Running the baseline

1. Change into the `baseline_llava` directory:
   ```bash
   cd baseline_llava
   ```
2. Run the demo script:
   ```bash
   python run_baseline_demo.py
   ```
   This will print the attention matrix for one head, the logits for
   the last token, and a sequence of generated token IDs.

## Exploring the SECOND hooks

1
. Change into the `second` directory:
   ```bash
   cd second
   ```
2. Run the demo:
   ```bash
   python run_second_demo.py
   ```
   The script will apply the positional embedding interpolation,
   attention top‑k masking and contrastive mixing functions to dummy
   data and print the results. To integrate these hooks with a real
   vision‑language model you will need to adapt the `run_second`

   function to

    the API of that model.

## Grok‑1 example

The `grok` directory contains `run.py`, copied from the
`grok-1-20240322` repository. That script demonstrates how to build
and run the Grok‑1 model using JAX and Haiku. The full implementation
of Grok‑1 lives outside this project and is not included here. If you
wish to experiment with Grok‑1 you should clone the original
repository and install its dependencies.

## Notes

- The baseline model provided here is intentionally small and
  educational. It cannot replace a full LLaVA model. However, its
  simplicity makes it easy to understand and extend.

  `run_second_demo.py`, install PyTorch in your environment.
- The Grok example is included for completeness but is not executable
  without the rest of the Grok codebase. Treat it as a template.


## Combining LLaVA and SECOND

At present the code in this repository does not implement a working integration between the Mini‑LLaVA baseline and the SECOND hooks. The baseline in `baseline_llava` is a pure NumPy mini‑Transformer for demonstration purposes, whereas the SECOND hooks in `second/second_hooks.py` are implemented in PyTorch and expect to operate on PyTorch tensors from a real LLaVA model. As a result, there is no out‑of‑the‑box way to run SECOND on top of the baseline model. When we attempted to run the SECOND demo script in this environment it failed with a missing PyTorch dependency.

To use SECOND with a proper LLaVA model you need to:

1. Install a PyTorch version of LLaVA and ensure you have an up‑to‑date vision‑language model checkpoint (e.g. LLaVA 1.5 or LLaVA‑NeXT).
2. Install PyTorch in your environment. The demo scripts here will not run without it.
3. Use the helper functions in `second/second_hooks.py` within the forward pass of your LLaVA model. After encoding an image with the vision tower, apply positional embedding interpolation if the stage resolution differs from the ViT’s training grid; use `topk_mask_from_attn` on the attention map to select the most relevant visual tokens; and use `contrastive_mix` to blend logits from multiple scales.
own model’s API.

Until these steps are carried out, the baseline and SECOND remain separate demonstrations rather than an integrated pipeline.
