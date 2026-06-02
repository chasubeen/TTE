# data_provider/build_dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from .load_dataset import get_data_transforms, load_mvtec_paths, load_realiad_paths


class MVTecDataset(Dataset):
    def __init__(self, root, phase, img_size, center_size,
                 mean_train=None, std_train=None):
        self.data_transform, self.gt_transform = get_data_transforms(
            img_size, center_size, mean_train, std_train
        )
        (self.img_paths,
         self.gt_paths,
         self.labels,
         self.types) = load_mvtec_paths(root, phase)
        self.phase = phase

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label    = int(self.labels[idx])
        gt_path  = self.gt_paths[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.data_transform(img)

        if self.phase == "train" or label == 0:
            _, H, W = img.size()
            gt = torch.zeros(1, H, W)
        else:
            gt_img = Image.open(gt_path)
            gt = self.gt_transform(gt_img)

        if img.size()[1:] != gt.size()[1:]:
            raise ValueError("image.size != gt.size")

        if self.phase == "train":
            # Return (img, label) only during training since gt is not needed
            return img, label
        return img, gt, label, img_path


class RealIADDataset(Dataset):
    def __init__(self, root, category, phase, img_size, center_size,
                 mean_train=None, std_train=None):
        self.data_transform, self.gt_transform = get_data_transforms(
            img_size, center_size, mean_train, std_train
        )
        (self.img_paths,
         self.gt_paths,
         self.labels,
         self.types) = load_realiad_paths(root, category, phase)
        self.phase = phase

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label    = int(self.labels[idx])
        gt_path  = self.gt_paths[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.data_transform(img)

        if self.phase == "train":
            return img, label  # RealIAD train does not use gt

        if label == 0 or gt_path is None:
            gt = torch.zeros([1, img.size(-2), img.size(-2)])
        else:
            gt_img = Image.open(gt_path)
            gt = self.gt_transform(gt_img)

        if img.size()[1:] != gt.size()[1:]:
            raise ValueError("image.size != gt.size")
        return img, gt, label, img_path


## dataloader builders
def build_dataloaders_mvtec(
    root, phase, img_size, center_size,
    batch_size, num_workers=4, shuffle=True,
):
    dataset = MVTecDataset(
        root=root,
        phase=phase,
        img_size=img_size,
        center_size=center_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if phase == "train" else False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataset, loader


def build_dataloaders_realiad(
    root, category, phase, img_size, center_size,
    batch_size, num_workers=4, shuffle=True,
):
    dataset = RealIADDataset(
        root=root,
        category=category,
        phase=phase,
        img_size=img_size,
        center_size=center_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if phase == "train" else False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataset, loader