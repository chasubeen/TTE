"""Backward-compatibility shim for src.utils.metrics.

The slim canonical implementation now lives in `src.utils.eval_metrics`.

Existing analysis/measure_*/phase*/rq* scripts that import:
    from src.utils.metrics import ader_evaluator
    from src.utils.metrics import f1_score_max
will continue to work via this shim.

Legacy helpers (memory_patch_scores, penalty_patch_scores,
dual_memory_patch_scores, scores_list_to_maps, image_level_scores_from_maps,
resize_gt_masks, get_logger, setup_seed) are no longer provided here.
If reproducing src10/src11-era experiments requires them, copy from the
git history or from src11/utils/metrics.py.
"""
from src.utils.eval_metrics import ader_evaluator, f1_score_max

__all__ = ["ader_evaluator", "f1_score_max"]
