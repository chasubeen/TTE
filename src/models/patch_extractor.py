from typing import List, Optional, Tuple

import os
import numpy as np
import torch
import torch.nn as nn

# --- backbones ---
from src.models.cnn_encoder import resolve_cnn_backbone # CNN-based
from src.models.vit_encoder import load as load_vit # ViT-based
from src.utils.aug_funcs import embedding_concat


class PatchFeatureExtractor(nn.Module):
    """
    Patch-level feature extractor.

    - ViT family:
        - Uses vit_encoder.load(backbone_name)
        - Extracts patch tokens via get_intermediate_layers
    - CNN family:
        - Uses cnn_encoder.resolve_cnn_backbone(backbone_name, out_indices)
        - Concatenates/aggregates feature maps specified by out_indices

    Args:
        backbone_name: str
            ViT: e.g. "dinov2reg_vit_base_14", "dino_vit_small_16"
            CNN: preset name (e.g. "wrn50_2_l2l3") or timm name (e.g. "wide_resnet50_2")
        is_vit_like: bool
            True  -> use vit_encoder
            False -> use cnn_encoder + timm features_only
        n_layers: int
            ViT: number of layers to retrieve from get_intermediate_layers
        layer_ids: List[int] | None
            ViT: explicit layer indices (e.g. [5,11,17,23]).
            None uses the last n_layers blocks.
        cnn_out_indices: List[int] | None
            CNN: out_indices for timm features_only.
            None falls back to preset default or [-1] (last feature map).
        agg: str
            ViT: aggregation over multi-layer tokens -> "mean" / "concat"
            CNN: used for multi-scale feature map concatenation (embedding_concat)
        img_size: int
            Input size for dummy forward to infer patch grid dimensions.
    """
    def __init__(
        self,
        backbone_name: str,
        is_vit_like: bool = True,
        n_layers: int = 1,
        layer_ids: Optional[List[int]] = None,
        cnn_out_indices: Optional[List[int]] = None,
        agg: str = "mean",
        img_size: int = 518,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.is_vit_like = is_vit_like
        self.n_layers = n_layers
        self.layer_ids = layer_ids
        self.cnn_out_indices = cnn_out_indices
        self.agg = agg
        self.img_size = img_size
        
        if self.is_vit_like:
            # ViT backbone
            if backbone_name.startswith("dinov2_") and "reg" not in backbone_name:
                # Use torch.hub for standard DINOv2 (AnomalyDINO-compatible)
                self.backbone = self._load_dinov2_hub(backbone_name)
            else:
                self.backbone = load_vit(backbone_name)
            self._init_vit()
        else:
            # CNN backbone: preset + timm
            self.backbone, self.cnn_out_indices = resolve_cnn_backbone(
                backbone_name,
                out_indices=self.cnn_out_indices,
                pretrained=True,
            )
            self._init_cnn()
        
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
    
    
    def _load_dinov2_hub(self, backbone_name: str):
        """Load DINOv2 via torch.hub (identical to AnomalyDINO official code)."""
        import torch
        # Map our naming to torch.hub model names
        # e.g. "dinov2_vit_small_14" -> "dinov2_vits14"
        parts = backbone_name.split("_")  # ["dinov2", "vit", "small", "14"]
        arch = parts[2]   # "small" / "base" / "large"
        ps = parts[3]     # "14"
        hub_name = f"dinov2_vit{arch[0]}{ps}"  # "dinov2_vits14"
        hub_cache = os.path.expanduser('~/.cache/torch/hub/facebookresearch_dinov2_main')
        if os.path.exists(hub_cache):
            model = torch.hub.load(hub_cache, hub_name, source='local')
        else:
            model = torch.hub.load('facebookresearch/dinov2', hub_name)
        model.eval()
        self._use_hub_dinov2 = True  # Hub's get_intermediate_layers returns patch tokens only (no CLS)
        return model

    ## backbone initialization
    def _init_vit(self):
        """Lazy initialization for ViT: infer patch grid size via dummy forward."""
        # Hub DINOv2: get_intermediate_layers already strips CLS/REG tokens
        # Local DINOv2: returns CLS + REG + patch tokens, need manual stripping
        if getattr(self, '_use_hub_dinov2', False):
            self.num_prefix_tokens = 0
        else:
            self.num_prefix_tokens = 1 + getattr(self.backbone, 'num_register_tokens', 0)  # CLS + REG

        dummy = torch.zeros(1, 3, self.img_size, self.img_size)
        with torch.no_grad():
            out = self.backbone.get_intermediate_layers(dummy, n=1)[0]  # (1, N, C) or (1, 1+REG+N, C)
        N = out.shape[1] - self.num_prefix_tokens
        side = int(round(np.sqrt(N)))
        self.patch_grid = (side, side)
        self.feature_dim = int(out.shape[-1])
    
    def _init_cnn(self):
        """Infer feature dim and spatial size from CNN features_only backbone.
        Applies embedding_concat (same as forward) to compute actual feature_dim.
        """
        dummy = torch.zeros(1, 3, self.img_size, self.img_size)
        with torch.no_grad():
            feats = self.backbone(dummy)  # list of feature maps per out_indices

        if isinstance(feats, (list, tuple)) and len(feats) > 1:
            # Apply embedding_concat (same as forward)
            fm = feats[0]
            for f in feats[1:]:
                fm = embedding_concat(fm, f)
        elif isinstance(feats, (list, tuple)):
            fm = feats[0]
        else:
            fm = feats

        _, C, H, W = fm.shape
        self.patch_grid = (H, W)
        self.feature_dim = int(C)


    @torch.no_grad()
    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
            x: (B, 3, H, W) normalized image tensor.
        Returns:
            patch_tok: (B, N_patches, C)
            spatial_shape: (H_p, W_p)
        """
        if self.is_vit_like:
            return self._forward_vit(x)
        else:
            return self._forward_cnn(x)
    
    
    def _forward_vit(self, x: torch.Tensor):
        H_p, W_p = self.patch_grid

        # Use explicit layer_ids if provided, otherwise use n_layers
        if self.layer_ids is not None:
            feats = self.backbone.get_intermediate_layers(x, n=self.layer_ids)
        else:
            feats = self.backbone.get_intermediate_layers(x, n=self.n_layers)
        
        if isinstance(feats, tuple):
            feats = list(feats)
        
        # Extract patch tokens only (exclude CLS + register tokens), then aggregate
        patch_list = []
        for f in feats:
            patch_list.append(f[:, self.num_prefix_tokens:, :])  # (B, N_patches, C)
        
        if len(patch_list) == 1:
            patch_tok = patch_list[0] # (B, N, C)
        else:
            if self.agg == "mean":
                patch_tok = torch.stack(patch_list, dim=0).mean(0) # (B, N, C)
            elif self.agg == "concat":
                patch_tok = torch.cat(patch_list, dim=-1) # (B, N, C*L)
            else:
                raise ValueError(f"Unknown agg: {self.agg}")
        
        return patch_tok, self.patch_grid


    def _forward_cnn(self, x: torch.Tensor):
        """
        timm features_only model.
        Returns feature maps for the specified out_indices.
        e.g. wide_resnet50_2 + out_indices=[2,3] -> layer2, layer3 feature maps.
        """
        feats_list = self.backbone(x)  # list[Tensor], each (B, C, H, W)

        # Concatenate multi-scale feature maps via embedding_concat
        if len(feats_list) == 1:
            fm = feats_list[0]
        else:
            fm = feats_list[0]
            for f in feats_list[1:]:
                fm = embedding_concat(fm, f)  # (B, C_concat, H_high, W_high)
        
        B, C, H_p, W_p = fm.shape
        patch_tok = fm.permute(0, 2, 3, 1).contiguous().view(B, H_p * W_p, C)
        
        return patch_tok, (H_p, W_p)
