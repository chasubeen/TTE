"""Backward-compatibility shim — moved to `src.memory.build`.

M₀ construction (and test-feature loading) now lives in the `memory` package.
Existing imports `from src.pipeline.memory_build import build_memory, ...`
keep working.
"""
from src.memory.build import build_memory, load_test_images

__all__ = ["build_memory", "load_test_images"]
