"""Phase 1 — end-to-end generation benchmark.

Measures the four numbers that go in BENCHMARKS.md and on the resume:
  - tokens/sec   : decode throughput (new tokens / total decode time)
  - TTFT         : time to first token (prefill latency, felt by the user)
  - p50 / p99    : per-request latency distribution

This is the bar Phase 2 has to beat. The metric math (percentiles, aggregation)
is done for you; the TODO(you) is wiring in the generation calls once the loop
in engine/generate.py exists.

Run (on a CUDA machine, repo root):  python bench/run.py --n 20 --max-new 128
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

# Make `import engine` work when run as a plain script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from engine import ModelConfig, load_model
from engine.generate import generate
from engine.sampling import SamplingParams


@dataclass
class RunResult:
    ttft_s: float       # time to first token
    total_s: float      # full request wall time
    n_tokens: int       # tokens actually generated


def time_one_request(model, prompt: str, max_new_tokens: int, params) -> RunResult:
    """Drive one generation, timing first-token and total latency.

    `generate` is a generator, so we can timestamp the *first* yielded token
    (TTFT) separately from the rest — that split is the whole point.
    """
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    ttft = None
    n = 0

    # TODO(you): iterate generate(...) and:
    #   - record `ttft = time.perf_counter() - start` on the first token
    #   - count tokens
    # for _ in generate(model, prompt, max_new_tokens, params):
    #     if ttft is None:
    #         ttft = time.perf_counter() - start
    #     n += 1
    raise NotImplementedError("wire up the generate() loop, then delete this line")

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total = time.perf_counter() - start
    return RunResult(ttft_s=ttft or total, total_s=total, n_tokens=n)


def percentile(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank percentile (p in [0, 100]). Input must be pre-sorted."""
    if not sorted_vals:
        return float("nan")
    k = max(0, min(len(sorted_vals) - 1, round(p / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--prompt", default="The future of GPU computing is")
    ap.add_argument("--n", type=int, default=20, help="number of timed requests")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    model = load_model(ModelConfig(name=args.model))
    params = SamplingParams(temperature=args.temperature)

    # One warmup request so JIT/cudnn autotune/clocks don't pollute the numbers.
    time_one_request(model, args.prompt, args.max_new, params)

    results = [
        time_one_request(model, args.prompt, args.max_new, params)
        for _ in range(args.n)
    ]

    total_tokens = sum(r.n_tokens for r in results)
    decode_time = sum(r.total_s - r.ttft_s for r in results)
    tok_per_s = total_tokens / decode_time if decode_time else float("nan")

    ttfts = sorted(r.ttft_s for r in results)
    totals = sorted(r.total_s for r in results)

    print(f"model={args.model}  requests={args.n}  max_new={args.max_new}")
    print(f"  throughput : {tok_per_s:8.1f} tok/s")
    print(f"  TTFT  p50  : {percentile(ttfts, 50) * 1e3:8.1f} ms")
    print(f"  latency p50: {percentile(totals, 50) * 1e3:8.1f} ms")
    print(f"  latency p99: {percentile(totals, 99) * 1e3:8.1f} ms")


if __name__ == "__main__":
    main()
