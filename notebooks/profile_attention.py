"""Driver for Nsight profiling (see NSIGHT.md).

Two modes:
  --impl {naive,triton} [--seq N]   : hammer the attention op in a loop so ncu can
                                       capture its kernels.
  --generate                        : run a short GPT-2 generation (optionally with
                                       the Triton kernel monkeypatched in) so nsys
                                       can trace the whole decode path.

Run on a rented GPU with Nsight (Colab blocks ncu perf counters).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from kernels.triton.fused_attention import fused_attention_bhsd


def naive_attention(q, k, v, causal=True):
    scale = q.shape[-1] ** -0.5
    s = (q @ k.transpose(-1, -2)) * scale
    if causal:
        M, N = q.shape[-2], k.shape[-2]
        mask = torch.ones(M, N, device=q.device, dtype=torch.bool).tril(N - M)
        s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s.float(), -1).to(q.dtype)
    return p @ v


def _triton_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, scale=None, **kw):
    """Route SDPA through our kernel when it's safe; else fall back (stays correct)."""
    if (attn_mask is None and query.dtype in (torch.float16, torch.bfloat16)
            and query.shape[-1] == key.shape[-1]):
        return fused_attention_bhsd(query, key, value, causal=bool(is_causal))
    return _triton_sdpa._orig(query, key, value, attn_mask=attn_mask,
                              dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["naive", "triton"], default="triton")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--model", default="gpt2")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU"

    if args.generate:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        impl = "sdpa" if args.impl == "triton" else "eager"
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, attn_implementation=impl).cuda().eval()
        if args.impl == "triton":
            _triton_sdpa._orig = F.scaled_dot_product_attention
            F.scaled_dot_product_attention = _triton_sdpa
        ids = tok("The future of GPU computing is", return_tensors="pt").input_ids.cuda()
        with torch.inference_mode():
            for _ in range(4):
                model.generate(ids, max_new_tokens=64, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        return

    B, H, D = 1, 8, 64
    q = torch.randn(B, H, args.seq, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, args.seq, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, args.seq, D, device="cuda", dtype=torch.float16)
    fn = naive_attention if args.impl == "naive" else fused_attention_bhsd
    with torch.inference_mode():
        for _ in range(args.iters):
            fn(q, k, v, True)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
