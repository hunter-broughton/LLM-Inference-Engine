"""Mini LLM inference engine.

Phase 1 (this package, baseline): model loading + a plain generation loop.
Phase 2 adds the paged KV cache (kv_cache.py) and continuous-batching scheduler
(scheduler.py). Public surface is intentionally tiny — import what you need:

    from engine import load_model, generate
"""

from .generate import generate
from .model import ModelConfig, load_model

__all__ = ["load_model", "ModelConfig", "generate"]
