from __future__ import annotations
from typing import List, Optional, Tuple

import timm
import torch.nn as nn


"""
CNN backbone loader.
- Serves the same role as vit_encoder.load, but for CNN backbones.
- Maps preset names to timm model names + out_indices.
"""

### Extend as needed
_CNN_PRESETS = {
    # PatchCore-style: Wide ResNet-50-2, using layer2 + layer3 features
    "wrn50_2_l2l3": {
        "timm_name": "wide_resnet50_2",
        "out_indices": [2, 3],   # layer2, layer3
    },
    # Example: use only the last feature map of wide_resnet50_2
    # "wrn50_2_last": {
    #     "timm_name": "wide_resnet50_2",
    #     "out_indices": [-1],
    # }
}


def resolve_cnn_backbone(
    name: str,
    out_indices: Optional[List[int]] = None,
    pretrained: bool = True,
    **timm_kwargs,
) -> Tuple[nn.Module, List[int]]:
    """
    Resolve a CNN backbone by name.
    - If name matches a preset: use the preset's timm_name / out_indices.
    - Otherwise: treat name as a timm model name, using the provided out_indices
      (defaults to [-1], i.e. last feature map only).

    Returns:
        model: timm features_only model.
        out_indices: the out_indices actually used (for reference by PatchFeatureExtractor).
    """
    if name in _CNN_PRESETS:
        cfg = _CNN_PRESETS[name]
        timm_name = cfg["timm_name"]
        if out_indices is None:
            out_indices = cfg["out_indices"]
    else:
        # Not a preset; treat as a timm model name directly
        timm_name = name
        if out_indices is None:
            out_indices = [-1]

    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        features_only=True,
        out_indices=tuple(out_indices),
        **timm_kwargs,
    )
    return model, out_indices