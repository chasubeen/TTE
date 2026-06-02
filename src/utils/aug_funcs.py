"""
Feature utility functions for multi-scale feature aggregation.
"""
import torch
import torch.nn.functional as F


def embedding_concat(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Spatially align two feature maps and concatenate along the channel dimension.

    For the same spatial position (same patch region), concatenate features
    from different scales/layers along the channel dimension. Used for
    multi-layer/multi-scale concatenation.

    Args:
        x: (B, C1, H1, W1) high-resolution feature map
        y: (B, C2, H2, W2) low-resolution feature map, H2 = H1 / s, W2 = W1 / s

    Returns:
        (B, C1+C2, H1, W1) concatenated feature map
    """
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()

    if H1 % H2 != 0 or W1 % W2 != 0:
        raise ValueError(
            f"embedding_concat: spatial sizes not divisible: (H1,W1)=({H1},{W1}), (H2,W2)=({H2},{W2})"
        )
    s = H1 // H2

    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)  # (B, C1*s*s, H2*W2)
    x = x.view(B, C1, s*s, H2, W2)  # (B, C1, s*s, H2, W2)

    y_expanded = y.unsqueeze(2).expand(B, C2, s*s, H2, W2)  # (B, C2, s*s, H2, W2)

    z = torch.cat([x, y_expanded], dim=1)  # (B, C1+C2, s*s, H2, W2)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z
