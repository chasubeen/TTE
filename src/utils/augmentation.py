"""
Augmentation utilities for few-shot anomaly detection.
Based on AnomalyDINO (WACV'25) official code: https://github.com/dammsi/AnomalyDINO

Two augmentation strategies:
  1. Rotation augmentation: rotate reference images to diversify memory bank
  2. PCA-based background masking: remove background patches from memory bank
"""

import numpy as np
import cv2
from sklearn.decomposition import PCA
from typing import List, Optional, Tuple


# ============================================================
# Rotation Augmentation (from AnomalyDINO src/utils.py)
# ============================================================

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by given angle (degrees) around center."""
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1],
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_DEFAULT,
    )
    return result


def augment_images_rotation(
    images: List[np.ndarray],
    angles: List[float] = [0, 45, 90, 135, 180, 225, 270, 315],
) -> List[np.ndarray]:
    """
    Apply rotation augmentation to a list of images.

    Args:
        images: list of (H, W, 3) numpy arrays (RGB)
        angles: rotation angles in degrees

    Returns:
        augmented: list of all rotated images (len = len(images) * len(angles))
    """
    augmented = []
    for img in images:
        for angle in angles:
            augmented.append(rotate_image(img, angle))
    return augmented


# ============================================================
# PCA-Based Background Masking (from AnomalyDINO src/backbones.py)
# ============================================================

def compute_foreground_mask(
    patch_features: np.ndarray,
    grid_size: Tuple[int, int],
    threshold: float = 10.0,
    kernel_size: int = 3,
    border: float = 0.2,
) -> np.ndarray:
    """
    Compute foreground mask using PCA on patch features.

    Uses the 1st principal component to separate foreground from background.
    Includes adaptive polarity check and morphological post-processing.

    Args:
        patch_features: (N_patches, D) feature array
        grid_size: (H_grid, W_grid) spatial grid dimensions
        threshold: PCA score threshold for foreground/background separation
        kernel_size: morphological kernel size (odd number)
        border: fraction of border to check for adaptive polarity

    Returns:
        mask: (N_patches,) boolean array — True = foreground (keep)
    """
    pca = PCA(n_components=1, svd_solver='randomized')
    first_pc = pca.fit_transform(patch_features.astype(np.float32))

    mask = first_pc > threshold

    # Adaptive polarity check: if center crop is mostly masked, flip polarity
    m = mask.reshape(grid_size)[
        int(grid_size[0] * border):int(grid_size[0] * (1 - border)),
        int(grid_size[1] * border):int(grid_size[1] * (1 - border)),
    ]
    if m.sum() <= m.size * 0.35:
        mask = -first_pc > threshold

    # Morphological post-processing: dilate + close to fill holes
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    return mask.squeeze()


# ============================================================
# Per-category masking/rotation defaults (from AnomalyDINO)
# ============================================================

# AnomalyDINO "agnostic" preset: all categories get rotation
# Masking applied per-category based on object type
MVTEC_MASKING_DEFAULT = {
    "bottle": False, "cable": False, "capsule": True, "carpet": False,
    "grid": False, "hazelnut": True, "leather": False, "metal_nut": False,
    "pill": True, "screw": True, "tile": False, "toothbrush": True,
    "transistor": False, "wood": False, "zipper": False,
}

VISA_MASKING_DEFAULT = {
    "candle": True, "capsules": True, "cashew": True, "chewinggum": True,
    "fryum": True, "macaroni1": True, "macaroni2": True,
    "pcb1": True, "pcb2": True, "pcb3": True, "pcb4": True, "pipe_fryum": True,
}


def get_masking_default(dataset_name: str, category: str) -> bool:
    """Get AnomalyDINO default masking setting for a category."""
    if dataset_name == "MVTecAD":
        return MVTEC_MASKING_DEFAULT.get(category, False)
    elif dataset_name in ("VisA", "VisA_pytorch"):
        return VISA_MASKING_DEFAULT.get(category, True)
    return False
