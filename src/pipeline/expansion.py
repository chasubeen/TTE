"""Backward-compatibility shim — moved to `src.memory.expander`.

Test-time selective memory expansion now lives in the `memory` package.
Existing imports `from src.pipeline.expansion import SelectiveExpander`
keep working (graveyard absorption variants — Plan v4 / Phase 3 / CBR / m2b' —
were removed; only the canonical reservoir/append paths remain).
"""
from src.memory.expander import (
    MemoryExpander,
    SelectiveExpander,
    DualMemoryExpander,
)

__all__ = ["MemoryExpander", "SelectiveExpander", "DualMemoryExpander"]
