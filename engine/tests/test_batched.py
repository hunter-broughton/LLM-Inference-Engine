"""Correctness of batched decode (Phase 2 throughput path).

The throughput number is only meaningful if batched decode is *exact*: batching
must not change what tokens come out. We assert batched greedy == single-stream
greedy, token for token, including for prompts of DIFFERENT lengths — the case
that left-padding + position_ids has to get right.

Run:  pytest engine/tests/test_batched.py
"""

import pytest

from engine.batched import generate_batched
from engine.generate import generate
from engine.model import ModelConfig, load_model
from engine.sampling import SamplingParams


@pytest.fixture(scope="module")
def model():
    return load_model(ModelConfig(name="gpt2", device="cpu"))


def test_batched_greedy_matches_single_stream(model):
    prompts = [
        "The future of GPU computing is",       # 6 tokens
        "Once upon a time",                      # 4 tokens
        "In a shocking finding, scientists discovered that",  # longer
    ]
    g = SamplingParams(temperature=0.0)

    batched = generate_batched(model, prompts, max_new_tokens=20, params=g)
    single = [list(generate(model, p, max_new_tokens=20, params=g)) for p in prompts]

    for i, p in enumerate(prompts):
        assert batched[i] == single[i], f"batched != single-stream for prompt {i!r}"


def test_batched_shapes(model):
    prompts = ["Hello world", "A B C", "One two three four"]
    outs = generate_batched(model, prompts, max_new_tokens=10,
                            params=SamplingParams(temperature=0.0))
    assert len(outs) == 3
    assert all(len(o) <= 10 for o in outs)
