"""Memory bank construction (training phase).

Builds the initial memory bank M₀ from few-shot support images using
frozen DINOv2 features and 8-fold rotation augmentation (AnomalyDINO style).
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms as T

from src.data_provider import create_dataloaders
from src.utils.augmentation import augment_images_rotation
from src.configs import IMG_SIZE, ROTATION_ANGLES as DEFAULT_ROTATION_ANGLES, make_memory_bank


_TRANSFORM = T.Compose([
    T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_memory(category, extractor, device, data_path, dataset, shot=1, seed=1,
                 rotation_angles=None):
    """Build M₀ for one (category, shot, seed) configuration.

    Args:
        rotation_angles: optional list of rotation angles for M0 construction.
            None keeps the YAML default (8x). [] disables rotation augmentation.

    Returns:
        bank: MemoryBank fitted with raw + optional rotation-augmented features
        spatial_shape: (H, W) patch grid shape
    """
    data_root = str(Path(data_path) / dataset / category)
    train_dataset, _ = create_dataloaders(
        dataset_name=dataset, root=data_root, phase="train",
        img_size=IMG_SIZE, center_size=IMG_SIZE, batch_size=4, num_workers=0, shuffle=False,
    )

    # Few-shot sampling: contiguous block per seed
    n_total = len(train_dataset)
    start = seed * shot
    end = start + shot
    if end > n_total:
        start, end = 0, shot

    # ---- Raw features ----
    raw_feats, spatial_shape = [], None
    for idx in range(start, min(end, n_total)):
        img, _ = train_dataset[idx]
        with torch.no_grad():
            feats, sp = extractor(img.unsqueeze(0).to(device))
        raw_feats.append(feats.squeeze(0).cpu().numpy())
        spatial_shape = sp
    raw_np = np.concatenate(raw_feats, axis=0).astype(np.float32)

    # ---- Optional rotation augmentation features ----
    if rotation_angles is None:
        rotation_angles = DEFAULT_ROTATION_ANGLES
    rotation_angles = list(rotation_angles)

    chunks = [raw_np]
    if rotation_angles:
        train_dir = Path(data_path) / dataset / category / "train" / "good"
        img_files = sorted(train_dir.glob("*.*"))
        if shot < len(img_files):
            img_files = [img_files[i] for i in range(start, min(end, len(img_files)))]
        raw_images = [cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                      for p in img_files]
        aug_images = augment_images_rotation(raw_images, angles=rotation_angles)

        aug_feats = []
        for aug_img in aug_images:
            img_t = _TRANSFORM(Image.fromarray(aug_img)).unsqueeze(0)
            with torch.no_grad():
                feats, _ = extractor(img_t.to(device))
            aug_feats.append(feats.squeeze(0).cpu().numpy())
        if aug_feats:
            chunks.append(np.concatenate(aug_feats, axis=0).astype(np.float32))

    # ---- Combined memory bank ----
    full_np = np.concatenate(chunks, axis=0).astype(np.float32)
    bank = make_memory_bank(device)
    bank.fit(full_np)
    return bank, spatial_shape


def load_test_images(category, extractor, device, data_path, dataset):
    """Extract DINOv2 features for all test images of a category.

    Returns:
        list of dicts {feats: (N, D), label: int, gt_mask: (H, W)}
    """
    data_root = str(Path(data_path) / dataset / category)
    test_dataset, _ = create_dataloaders(
        dataset_name=dataset, root=data_root, phase="test",
        img_size=IMG_SIZE, center_size=IMG_SIZE, batch_size=4, num_workers=0, shuffle=False,
    )
    images = []
    for idx in range(len(test_dataset)):
        img, gt_mask, label, _ = test_dataset[idx]
        with torch.no_grad():
            feats, _ = extractor(img.unsqueeze(0).to(device))
        feats_np = feats.squeeze(0).cpu().numpy()
        gt_np = gt_mask.numpy().squeeze()
        if gt_np.ndim < 2:
            gt_np = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        images.append({"feats": feats_np, "label": int(label), "gt_mask": gt_np})
    return images
