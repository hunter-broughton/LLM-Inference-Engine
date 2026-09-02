"""Phase 2 — concurrency throughput: engine (batched) vs PyTorch baseline.

The Phase 2 headline number. For each concurrency level C we serve C requests two
ways and compare aggregate tokens/sec:

  baseline : HF `model.generate()` run C times SEQUENTIALLY — the stock PyTorch
             way to serve requests, no cross-request batching.
  engine   : `generate_batched()` — all C sequences in one batched decode loop.

Aggregate throughput = total tokens generated / wall time. The speedup is the
engine's aggregate tok/s over the baseline's. Run this on a GPU for the headline;
on CPU it already shows the effect.

Run:  python bench/concurrency.py --concurrency 1,8,32 --max-new 64
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from engine import ModelConfig, load_model
from engine.batched import generate_batched
from engine.sampling import SamplingParams

PROMPT = "The future of GPU computing is"


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_baseline(model, C: int, max_new: int) -> tuple[int, float]:
    """Serve C requests one at a time with stock HF generate() — no batching."""
    tok, hf = model.tokenizer, model.model
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(model.config.device)
    _sync()
    t0 = time.perf_counter()
    total = 0
    with torch.inference_mode():
        for _ in range(C):
            out = hf.generate(ids, max_new_tokens=max_new, do_sample=False,
                              pad_token_id=tok.eos_token_id)
            total += out.shape[1] - ids.shape[1]
    _sync()
    return total, time.perf_counter() - t0


def bench_engine(model, C: int, max_new: int, params) -> tuple[int, float]:
    """Serve C requests together via the engine's batched decode."""
    prompts = [PROMPT] * C
    _sync()
    t0 = time.perf_counter()
    outs = generate_batched(model, prompts, max_new_tokens=max_new, params=params)
    _sync()
    return sum(len(o) for o in outs), time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--concurrency", default="1,8,32")
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    model = load_model(ModelConfig(name=args.model))
    params = SamplingParams(temperature=0.0)
    levels = [int(x) for x in args.concurrency.split(",")]

    # Warm up both paths so weight-load / autotune costs don't skew the first level.
    bench_baseline(model, 1, 4)
    bench_engine(model, 2, 4, params)

    dev = model.config.device
    print(f"model={args.model}  device={dev}  max_new={args.max_new}\n")
    print(f"{'C':>4}  {'baseline tok/s':>15}  {'engine tok/s':>13}  {'speedup':>8}")
    print("-" * 48)
    for C in levels:
        bt, bd = bench_baseline(model, C, args.max_new)
        et, ed = bench_engine(model, C, args.max_new, params)
        base_tps, eng_tps = bt / bd, et / ed
        print(f"{C:>4}  {base_tps:>15.1f}  {eng_tps:>13.1f}  {eng_tps / base_tps:>7.2f}x")


if __name__ == "__main__":
    main()
