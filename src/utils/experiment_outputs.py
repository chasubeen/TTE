"""Reusable experiment-output helpers for src.

Every memory-expansion experiment should emit, through these helpers, a
uniform pair of artifacts:

  1. save_metric_table(...)    -> CSV (Excel-openable, utf-8-sig) of the 7
                                  evaluation metrics, one row per (arm,
                                  category, seed), plus per-(dataset, arm)
                                  MEAN rows.
  2. save_coverage_figure(...) -> PNG of Coverage(M_t) vs. batch index: faint
                                  per-category lines + a bold dataset-mean
                                  line, one panel per dataset.

compute_coverage(...) is the shared measurement primitive — the fraction of
foreground-normal test patches within theta_base 1-NN distance of a memory
bank (the §1.2 Coverage metric, evaluable against any memory snapshot M_t).

Dependency-light: csv (stdlib) + matplotlib (lazy-imported, Agg backend).
"""
import csv
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# 7 canonical evaluation metrics — order matches src.configs.METRICS and the
# ader_evaluator return vector.
METRIC_KEYS = ["I-AUROC", "I-AP", "I-F1_max", "P-AUROC", "P-AP", "P-F1_max", "AUPRO"]
METRIC_DISPLAY = {
    "I-AUROC": "I-AUROC", "I-AP": "I-AP", "I-F1_max": "I-F1",
    "P-AUROC": "P-AUROC", "P-AP": "P-AP", "P-F1_max": "P-F1", "AUPRO": "AUPRO",
}
_ID_COLS = ("dataset", "arm", "category", "seed")


def compute_coverage(bank, fg_feats, theta_base):
    """Coverage(M) — fraction of fg_feats within theta_base 1-NN distance of `bank`.

    Args:
        bank:       a MemoryBank (M_t) exposing .query(feats, k).
        fg_feats:   (N, D) foreground-normal test patch features.
        theta_base: M0-derived coverage scale (mean intra-M0 NN distance).

    Returns:
        float in [0, 1].  Set Monotonicity guarantees this is non-decreasing as
        the bank grows, so a trajectory built from successive M_t must be
        monotone — a useful invariant for callers to assert.
    """
    fg_feats = np.asarray(fg_feats, dtype=np.float32)
    if len(fg_feats) == 0:
        return 0.0
    d, _ = bank.query(fg_feats, k=1)
    return float(np.mean(d[:, 0] < theta_base))


def save_metric_table(records, out_path):
    """Write a 7-metric summary CSV (Excel-openable, utf-8-sig).

    Args:
        records:  list of dicts.  Each dict carries identifier fields (any
                  subset of dataset/arm/category/seed) and the 7 metric values
                  keyed by METRIC_KEYS.
        out_path: destination .csv path.

    Per-(dataset, arm) MEAN rows are appended automatically when both 'dataset'
    and 'arm' identifiers are present.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        logger.warning("save_metric_table: no records — skipped.")
        return

    id_cols = [c for c in _ID_COLS if any(c in r for r in records)]
    header = id_cols + [METRIC_DISPLAY[m] for m in METRIC_KEYS]

    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else v

    def _row(rec):
        row = {c: rec.get(c, "") for c in id_cols}
        for m in METRIC_KEYS:
            row[METRIC_DISPLAY[m]] = _fmt(rec.get(m, ""))
        return row

    body = [_row(r) for r in records]

    mean_rows = []
    if "dataset" in id_cols and "arm" in id_cols:
        groups = {}
        for r in records:
            groups.setdefault((r.get("dataset", ""), r.get("arm", "")), []).append(r)
        for (ds, arm), recs in groups.items():
            mr = {c: "" for c in id_cols}
            mr["dataset"], mr["arm"] = ds, arm
            if "category" in id_cols:
                mr["category"] = "(mean)"
            if "seed" in id_cols:
                mr["seed"] = "(all)"
            for m in METRIC_KEYS:
                vals = [r[m] for r in recs
                        if isinstance(r.get(m), (int, float)) and not isinstance(r.get(m), bool)]
                mr[METRIC_DISPLAY[m]] = (f"{np.mean(vals):.4f}" if vals else "")
            mean_rows.append(mr)

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(body)
        w.writerows(mean_rows)
    logger.info(f"Metric table saved: {out_path}  "
                f"({len(body)} rows + {len(mean_rows)} mean)")


def save_coverage_figure(trajectories, out_path, title=None):
    """Line plot of Coverage(M_t) vs. batch index — one panel per dataset.

    Args:
        trajectories: {dataset: {category: [cov_batch0, cov_batch1, ...]}}.
                      Trajectories may differ in length across categories; the
                      dataset-mean line is taken over the common-length prefix.
        out_path:     destination .png path.
        title:        optional figure suptitle.

    Faint grey per-category lines + a bold dataset-mean line.  cov[0] is the
    pre-expansion M0 coverage.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    datasets = [d for d in trajectories if trajectories.get(d)]
    if not datasets:
        logger.warning("save_coverage_figure: no trajectories — skipped.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 4.3),
                             squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        cats = trajectories[ds]
        for cat, traj in sorted(cats.items()):
            ax.plot(range(len(traj)), traj, color="0.72", lw=0.9, alpha=0.85, zorder=1)
        lengths = [len(t) for t in cats.values() if len(t) > 0]
        if lengths:
            min_len = min(lengths)
            mean_traj = np.mean([t[:min_len] for t in cats.values() if len(t) >= min_len],
                                axis=0)
            ax.plot(range(min_len), mean_traj, color="C0", lw=2.6, marker="o", ms=4,
                    zorder=3, label=f"{ds} mean (n={len(cats)} cat)")
            ax.legend(loc="lower right", fontsize=9)
        ax.set_xlabel("test batch index")
        ax.set_ylabel("Coverage(M_t)")
        ax.set_title(ds)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Coverage figure saved: {out_path}")
