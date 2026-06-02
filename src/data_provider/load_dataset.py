import os
import glob
import json
import numpy as np
from PIL import Image
from torchvision import transforms


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train  = [0.229, 0.224, 0.225] if std_train is None else std_train

    data_transforms = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(isize),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train, std=std_train),
    ])
    gt_transforms = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(isize),
        transforms.ToTensor(),
    ])
    return data_transforms, gt_transforms


## MVTec-AD
def load_mvtec_paths(root, phase):
    """Collect and return all paths/labels for a single MVTec-AD class (root)."""
    if phase == "train":
        img_root = os.path.join(root, "train")
        gt_root  = None
    else:
        img_root = os.path.join(root, "test")
        gt_root  = os.path.join(root, "ground_truth")

    img_tot_paths, gt_tot_paths, labels, types = [], [], [], []

    defect_types = os.listdir(img_root)
    for defect_type in defect_types:
        patterns = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.bmp", "*.BMP"]
        img_paths = []
        for p in patterns:
            img_paths.extend(glob.glob(os.path.join(img_root, defect_type, p)))
        img_paths.sort()

        if defect_type == "good":
            img_tot_paths.extend(img_paths)
            gt_tot_paths.extend([0] * len(img_paths))   # dummy
            labels.extend([0] * len(img_paths))
            types.extend(["good"] * len(img_paths))
        else: 
            if gt_root is None:
                raise ValueError("gt_root must not be None for defect types other than 'good'")
            gt_paths = glob.glob(os.path.join(gt_root, defect_type, "*.png"))
            gt_paths.sort()
            img_tot_paths.extend(img_paths)
            gt_tot_paths.extend(gt_paths)
            labels.extend([1] * len(img_paths))
            types.extend([defect_type] * len(img_paths))

    if phase != "train":
        if len(img_tot_paths) != len(gt_tot_paths):
            raise ValueError("Something wrong with test and ground truth pair!")

    return (
        np.array(img_tot_paths),
        np.array(gt_tot_paths),
        np.array(labels),
        np.array(types),
    )


## RealIAD
def load_realiad_paths(root, category, phase):
    """Load paths/labels for a single RealIAD category."""
    img_root = os.path.join(root, "realiad_1024", category)

    json_path = os.path.join(root, "realiad_jsons", "realiad_jsons", category + ".json")
    with open(json_path) as f:
        class_json = json.load(f)

    img_paths, gt_paths, labels, types = [], [], [], []

    data_set = class_json[phase]
    for sample in data_set:
        img_paths.append(os.path.join(img_root, sample["image_path"]))
        label = (sample["anomaly_class"] != "OK")
        if label:
            gt_paths.append(os.path.join(img_root, sample["mask_path"]))
        else:
            gt_paths.append(None)
        labels.append(int(label))
        types.append(sample["anomaly_class"])

    return (
        np.array(img_paths),
        np.array(gt_paths, dtype=object),
        np.array(labels),
        np.array(types),
    )