"""Phase 1/2 — the PyTorch baseline, standalone.

This is the reference the engine is measured against: stock HuggingFace
`model.generate()`, single stream, no batching, no paging — the way a typical
PyTorch user serves a model. `bench/concurrency.py` compares the engine's batched
throughput to this; keeping it in its own file makes the comparison auditable.

Run:  python bench/baseline_hf.py --max-new 64
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from engine import ModelConfig, load_model

PROMPT = "The future of GPU computing is"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    model = load_model(ModelConfig(name=args.model))
    tok, hf = model.tokenizer, model.model
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(model.config.device)

    def one() -> int:
        out = hf.generate(ids, max_new_tokens=args.max_new, do_sample=False,
                          pad_token_id=tok.eos_token_id)
        return out.shape[1] - ids.shape[1]

    with torch.inference_mode():
        one()  # warmup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        total = sum(one() for _ in range(args.iters))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    print(f"model={args.model}  device={model.config.device}  "
          f"iters={args.iters}  max_new={args.max_new}")
    print(f"  PyTorch baseline (HF generate, single stream): {total / dt:.1f} tok/s")


if __name__ == "__main__":
    main()
