"""Memory expansion (Req-A / Selective Update).

Stateful expander driven inside the batch-by-batch test loop. Each batch's
confident-normal patches (chosen by the `Selector`, Req-B) are absorbed into the
memory under the configured policy:

  - reservoir : M₀ is frozen; a separate buffer B of capacity
                ``budget_B = (budget − 1)·|M₀|`` is filled then maintained by
                vectorized reservoir sampling (Algorithm R). The working bank is
                rebuilt as M₀ ∪ B each batch. (current canonical method)
  - append    : gap-priority append-only up to ``max_add`` (legacy default).

M₀ is always preserved; absorption only adds (Set Monotonicity holds), and the
gate that selects patches is memory-independent (anchored to M₀), so the
"Moving Normality Reference" drift cannot occur.
"""
import numpy as np

from src.configs import NOVELTY_RATIO, make_memory_bank
from src.selector import Selector


class MemoryExpander:
    """Test-time memory expander (Req-A). Owns the working bank + buffer."""

    def __init__(self, bank_init, estimator, cfg, device):
        self.estimator = estimator
        self.device = device
        self.cfg = cfg

        # Req-B gate (Selector): confident-normal = MLP recon error < τ_low.
        self.selector = Selector(estimator, cfg["tau_ratio_low"])
        self.tau_low = self.selector.tau_low

        self.max_add = int(bank_init.size() * (cfg["budget"] - 1.0))

        # Working memory M, initialized with M₀.
        self.bank = make_memory_bank(device)
        self.bank.fit(bank_init.features)

        self.added = 0
        self.n_absorbed_images = 0
        self.n_skipped_images = 0
        self.accepted_gap_scores = []     # float64 chunks, concatenated lazily
        self.rng = np.random.default_rng(int(cfg.get("insertion_seed", 0)))

        # mean intra-M₀ 1-NN distance — M₀-derived natural scale.
        if bank_init.size() >= 2:
            intra_dists, _ = bank_init.query(bank_init.features, k=2)
            self.theta_base = float(intra_dists[:, 1].mean())
        else:
            self.theta_base = 0.0

        # Optional novelty threshold (append-path ablation; 0 = off).
        novelty_ratio = cfg.get("novelty_ratio", NOVELTY_RATIO)
        self.theta_novel = self.theta_base * novelty_ratio if novelty_ratio > 0 else 0.0
        self.m0_size = bank_init.size()

        # Memory-management policy: "append" (legacy) or "reservoir" (current).
        self.memory_policy = cfg.get("memory_policy", "append")
        self._M0_feats = np.ascontiguousarray(bank_init.features.astype(np.float32))
        self.budget_B = self.max_add      # reservoir buffer capacity
        self.buffer = np.empty((0, self._M0_feats.shape[1]), dtype=np.float32)
        self.n_seen = 0                   # reservoir stream counter

    # --- backward-compatible aliases used by older diagnostics ---
    @property
    def bank_dom(self):
        return self.bank

    @property
    def bank_res(self):
        return None

    @property
    def bank_expanded(self):
        return self.bank

    def _order_candidates(self, z1, gap_scores):
        """Order accepted candidates before budgeted (append) insertion."""
        mode = self.cfg.get("insertion_mode", "gap_priority")
        if mode == "gap_priority":
            order = np.argsort(-gap_scores)
        elif mode == "random":
            order = self.rng.permutation(len(z1))
        elif mode == "stream_order":
            order = None
        else:
            raise ValueError(
                f"Unknown insertion_mode={mode!r}; expected "
                "'gap_priority', 'stream_order', or 'random'.")
        if order is not None:
            return z1[order], gap_scores[order]
        return z1, gap_scores

    def _gap_score_stats(self):
        if not self.accepted_gap_scores:
            return {"accepted_gap_mean": 0.0, "accepted_gap_p50": 0.0,
                    "accepted_gap_p90": 0.0, "accepted_gap_p95": 0.0,
                    "accepted_gap_count": 0}
        arr = np.concatenate(self.accepted_gap_scores)
        return {
            "accepted_gap_mean": float(arr.mean()),
            "accepted_gap_p50": float(np.percentile(arr, 50)),
            "accepted_gap_p90": float(np.percentile(arr, 90)),
            "accepted_gap_p95": float(np.percentile(arr, 95)),
            "accepted_gap_count": int(arr.size),
        }

    def absorb_batch(self, batch_feats):
        """Absorb this batch's confident-normal patches under the policy."""
        z1 = batch_feats[self.selector.confident_normal_mask(batch_feats)]

        if self.memory_policy == "reservoir":
            if len(z1) > 0:
                self._reservoir_absorb(np.ascontiguousarray(z1.astype(np.float32)))
            return

        # append (gap-priority) path
        if len(z1) == 0 or self.added >= self.max_add:
            return
        d_to_mem, _ = self.bank.query(z1, k=1)
        gap_scores = d_to_mem[:, 0]
        if self.theta_novel > 0:
            keep = gap_scores > self.theta_novel
            z1 = z1[keep]
            gap_scores = gap_scores[keep]
        if len(z1) == 0:
            return
        z1, gap_scores = self._order_candidates(z1, gap_scores)
        add_n = min(len(z1), self.max_add - self.added)
        self.bank.add(z1[:add_n], rebuild=True)
        self.accepted_gap_scores.append(gap_scores[:add_n].astype(np.float64, copy=False))
        self.added += add_n

    def _reservoir_absorb(self, z1):
        """Reservoir (vectorized Algorithm R) buffer maintenance.

        M₀ frozen; buffer B (capacity ``budget_B``) is filled then maintained by
        random replacement; the working bank is rebuilt as M₀ ∪ B each batch.
        Uses ``self.rng`` (seeded by ``insertion_seed``, default 0).
        """
        # Fill phase.
        if len(self.buffer) < self.budget_B:
            nf = min(self.budget_B - len(self.buffer), len(z1))
            self.buffer = np.concatenate([self.buffer, z1[:nf]], axis=0)
            self.n_seen += nf
            z1 = z1[nf:]
        # Replacement phase.
        if len(z1) > 0 and self.budget_B > 0:
            t = self.n_seen + np.arange(1, len(z1) + 1)
            j = (self.rng.random(len(z1)) * t).astype(np.int64)
            acc = j < self.budget_B
            if acc.any():
                self.buffer[j[acc]] = z1[acc]
            self.n_seen += len(z1)
        # Rebuild working bank = M₀ ∪ buffer.
        self.bank.fit(np.concatenate([self._M0_feats, self.buffer], axis=0))
        self.added = len(self.buffer)

    def stats(self):
        s = {
            "mem": self.bank.size(),
            "mem_dom": self.bank.size(),        # legacy alias
            "mem_res": 0,                        # legacy alias
            "mem_expanded": self.bank.size(),    # legacy alias
            "added": self.added,
            "added_res": 0,                      # legacy alias
            "budget_limit": self.max_add,
            "theta_base": self.theta_base,
            "theta_novel": self.theta_novel,
            "insertion_mode": self.cfg.get("insertion_mode", "gap_priority"),
            "memory_policy": self.memory_policy,
            "n_buffer": int(len(self.buffer)),
            "n_absorbed_images": self.n_absorbed_images,
            "n_skipped_images": self.n_skipped_images,
        }
        s.update(self._gap_score_stats())
        return s


# Backward-compatible names for existing imports.
SelectiveExpander = MemoryExpander
DualMemoryExpander = MemoryExpander
