"""PatchCore-style memory bank for src."""

from typing import Optional, Tuple, Union

import numpy as np
import torch

from src.memory.nn import FaissNN

ArrayLike = Union[np.ndarray, torch.Tensor]


class MemoryBank:
    """FAISS-backed memory bank for patch-level kNN anomaly scoring."""

    def __init__(
        self,
        device: Union[str, torch.device] = "cuda:0",
        faiss_on_gpu: bool = True,
        faiss_num_workers: int = 4,
        n_neighbors: int = 1,
        density_k: int = 5,
        metric: str = "l2",
    ):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        gpu_id = 0
        if "cuda" in str(self.device) and ":" in str(self.device):
            gpu_id = int(str(self.device).split(":")[-1])
        self.nn = FaissNN(
            on_gpu=faiss_on_gpu,
            num_workers=faiss_num_workers,
            device=gpu_id,
            metric=metric,
        )
        self.n_neighbors = n_neighbors
        self.density_k = density_k
        self.features: Optional[np.ndarray] = None
        self.local_density: Optional[np.ndarray] = None
        self.local_threshold: Optional[np.ndarray] = None

    @staticmethod
    def _to_numpy(x: ArrayLike) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    def fit(self, normal_features: ArrayLike) -> None:
        self.features = self._to_numpy(normal_features)
        self.nn.fit(self.features)
        self._update_local_density()

    def add(self, new_prototypes: ArrayLike, rebuild: bool = True, **_kwargs) -> int:
        newp = self._to_numpy(new_prototypes)
        if newp.ndim == 1:
            newp = newp[None, :]
        if len(newp) == 0:
            return self.size()

        prev = self.features if self.features is not None else np.empty((0, newp.shape[1]), dtype=np.float32)
        self.features = np.concatenate([prev, newp], axis=0).astype(np.float32)
        if rebuild:
            self.nn.fit(self.features)
            self._update_local_density()
        return self.size()

    def query(self, query_feats: ArrayLike, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if self.features is None:
            raise RuntimeError("MemoryBank is empty. Call .fit() or .add() first.")
        return self.nn.run(k or self.n_neighbors, self._to_numpy(query_feats))

    def remove(self, indices: np.ndarray) -> None:
        keep = np.ones(len(self.features), dtype=bool)
        keep[indices] = False
        self.features = self.features[keep]
        self.nn.fit(self.features)
        self._update_local_density()

    def size(self) -> int:
        return 0 if self.features is None else len(self.features)

    def _update_local_density(self) -> None:
        if self.features is None or len(self.features) < 2:
            self.local_density = None
            self.local_threshold = None
            return
        k = min(self.density_k, len(self.features) - 1)
        if k <= 0:
            self.local_density = None
            self.local_threshold = None
            return
        dists, _ = self.nn.run(k + 1, self.features)
        knn_dists = dists[:, 1:k + 1]
        self.local_density = knn_dists.mean(axis=1).astype(np.float32)
        self.local_threshold = np.median(knn_dists, axis=1).astype(np.float32)
