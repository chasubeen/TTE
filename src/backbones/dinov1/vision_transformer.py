"""Import-stub for the DINOv1 backbone.

The canonical src pipeline uses DINOv2 ViT-S/14 only; `models/vit_encoder.py`
imports this module at top level but never calls into it for the DINOv2 path.
These stubs exist solely to satisfy that import.  If a DINOv1 backbone is ever
selected, restore the real facebookresearch/dino `vision_transformer.py` here.
"""


def _unavailable(*_args, **_kwargs):
    raise NotImplementedError(
        "DINOv1 backbone is not vendored in src (only DINOv2 ViT-S/14 is). "
        "Drop facebookresearch/dino's vision_transformer.py here to enable it.")


# Accessed as vision_transformer.__dict__['vit_small'/'vit_base'] in the
# (unused) DINOv1 branch of vit_encoder.load.
vit_small = _unavailable
vit_base = _unavailable
vit_tiny = _unavailable
