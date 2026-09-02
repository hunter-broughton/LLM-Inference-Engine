"""Phase 2 — fused attention in Triton (FlashAttention-style).

Standard attention materializes the full [seq, seq] scores matrix in HBM — O(seq^2)
memory traffic and the dominant cost at long context, which makes it **memory-bound**.
FlashAttention never writes that matrix: it walks K/V in blocks, keeping a running
max and running sum (the *online softmax*) plus the output accumulator in registers,
so memory traffic drops to O(seq).

Performance notes (learned from profiling on a T4):
  - K is loaded already transposed ([D, BLOCK_N]) so the score matmul is a plain
    tl.dot with no in-kernel transpose.
  - @triton.autotune picks block sizes, warps, and — crucially — `num_stages`
    (software pipelining, which overlaps the K/V loads with the matmuls). Without
    pipelining the kernel is latency-bound on loads and loses to naive attention.

Correctness is gated against PyTorch (fp32 reference + SDPA) BEFORE any speedup is
reported — see the tests / the Colab notebook.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# Fixed launch config (NOT @triton.autotune keyed on N_KV). Autotuning keyed on the
# KV length is pathological in a decode loop: N_KV grows by one every token, so the
# autotuner re-benchmarks all configs at every step, every layer — measured ~14x
# SLOWER end-to-end. A single fixed config avoids that and gives the real win
# (+~17% end-to-end vs eager attention at long context; see BENCHMARKS.md).
BLOCK_M = 64
BLOCK_N = 64
NUM_WARPS = 4
NUM_STAGES = 2


@triton.jit
def _attention_kernel(
    Q, K, V, Out,
    scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    H, N_Q, N_KV,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Supports N_Q != N_KV (decode: 1 query vs a full KV cache). The queries are
    # the LAST N_Q positions of the sequence, so query row i has absolute position
    # q_offset + i, where q_offset = N_KV - N_Q. For prefill N_Q == N_KV (q_offset 0).
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    q_offset = N_KV - N_Q

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q_base = Q + b * stride_qb + h * stride_qh
    q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                mask=offs_m[:, None] < N_Q, other=0.0)   # [BLOCK_M, D]

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    # Causal: query at abs position (q_offset+m) attends keys <= that position.
    hi = q_offset + (pid_m + 1) * BLOCK_M if CAUSAL else N_KV
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kt = tl.load(k_base + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn,
                     mask=offs_n[None, :] < N_KV, other=0.0)   # [D, BLOCK_N], pre-transposed
        s = tl.dot(q, kt) * scale                             # [BLOCK_M, BLOCK_N]
        s = tl.where(offs_n[None, :] < N_KV, s, -float("inf"))
        if CAUSAL:
            s = tl.where((q_offset + offs_m[:, None]) >= offs_n[None, :], s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        corr = tl.exp(m_i - m_new)
        l_i = l_i * corr + tl.sum(p, 1)

        v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                    mask=offs_n[:, None] < N_KV, other=0.0)     # [BLOCK_N, D]
        acc = acc * corr[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_base = Out + b * stride_ob + h * stride_oh
    tl.store(o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
             acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_Q)


def fused_attention_bhsd(q, k, v, causal: bool = True):
    """Batched multi-head FlashAttention forward.

    q: [B, H, M, D], k/v: [B, H, N, D] with M <= N (M==N for prefill, M==1 for a
    single decode step against a length-N KV cache). Returns [B, H, M, D].
    """
    assert q.is_cuda and k.shape == v.shape
    assert q.shape[0] == k.shape[0] and q.shape[1] == k.shape[1] and q.shape[3] == k.shape[3]
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    B, H, M, D = q.shape
    N = k.shape[2]
    out = torch.empty_like(q)
    grid = (triton.cdiv(M, BLOCK_M), B * H)
    _attention_kernel[grid](
        q, k, v, out, D ** -0.5,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        H, M, N, D=D, CAUSAL=causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )
    return out


def fused_attention(q, k, v, causal: bool = True):
    """Single-head attention via the Triton kernel (the stub's entry point)."""
    assert q.is_cuda and q.shape == k.shape == v.shape
    out = fused_attention_bhsd(q[None, None], k[None, None], v[None, None], causal)
    return out[0, 0]
