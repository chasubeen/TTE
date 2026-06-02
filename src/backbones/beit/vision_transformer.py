"""Import-stub for the BEiT/BEiTv2 backbone.

`models/vit_encoder.py` imports `beitv2_base_patch16_448` and
`beitv2_base_patch16_224` at top level but never calls them for the canonical
DINOv2 ViT-S/14 path.  These stubs satisfy that import; restore the real BEiT
`vision_transformer.py` here if a BEiT backbone is ever selected.
"""


def _unavailable(*_args, **_kwargs):
    raise NotImplementedError(
        "BEiT backbone is not vendored in src (only DINOv2 ViT-S/14 is).")


def beitv2_base_patch16_448(*args, **kwargs):
    return _unavailable(*args, **kwargs)


def beitv2_base_patch16_224(*args, **kwargs):
    return _unavailable(*args, **kwargs)
