from src.data_provider.build_dataset import build_dataloaders_mvtec, build_dataloaders_realiad

def create_dataloaders(
    dataset_name,
    root,
    phase,
    img_size,
    center_size,
    batch_size,
    num_workers=4,
    shuffle=True,
    **extra_kwargs,
):
    """
    Call the appropriate dataloader builder based on dataset_name.
    extra_kwargs: additional arguments such as category for RealIAD.
    """
    if dataset_name in ("MVTecAD", "VisA", "VisA_pytorch"):
        return build_dataloaders_mvtec(
            root=root,
            phase=phase,
            img_size=img_size,
            center_size=center_size,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )
    elif dataset_name == "Real-IAD":
        category = extra_kwargs.get("category")
        if category is None:
            raise ValueError("RealIAD requires 'category' argument.")
        return build_dataloaders_realiad(
            root=root,
            category=category,
            phase=phase,
            img_size=img_size,
            center_size=center_size,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")