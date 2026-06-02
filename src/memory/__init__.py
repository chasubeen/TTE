"""memory — memory-bank management (Req-A / Selective Update).

  - `MemoryBank`     : FAISS-backed normal-patch bank (k-NN scoring substrate)
  - `MemoryExpander` : test-time selective expansion (reservoir / append)
  - `build_memory` / `load_test_images` live in `src.memory.build`
    (imported from there directly to avoid pulling the data/backbone stack here).
"""
from src.memory.bank import MemoryBank
from src.memory.expander import MemoryExpander, SelectiveExpander, DualMemoryExpander

__all__ = [
    "MemoryBank",
    "MemoryExpander",
    "SelectiveExpander",
    "DualMemoryExpander",
]
