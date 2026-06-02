"""FAISS nearest-neighbor wrapper used by src.memory.bank."""

from typing import Tuple

import faiss
import numpy as np
import torch


def _as_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return bool(int(x))
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y", "t")
    return bool(x)


class FaissNN:
    def __init__(
        self,
        on_gpu: bool = False,
        num_workers: int = 4,
        device=0,
        use_float16: bool = False,
        temp_index_on_cpu: bool = True,
        metric: str = "l2",
    ) -> None:
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu = _as_bool(on_gpu)
        if self.on_gpu:
            try:
                self.on_gpu = faiss.get_num_gpus() > 0
            except Exception:
                self.on_gpu = False

        self.device = int(device)
        self.search_index = None
        self.use_float16 = _as_bool(use_float16)
        self.temp_index_on_cpu = _as_bool(temp_index_on_cpu)
        self.metric = metric.lower()
        self._gpu_res = faiss.StandardGpuResources() if self.on_gpu else None

    def _gpu_cloner_options(self):
        opts = faiss.GpuClonerOptions()
        if hasattr(opts, "useFloat16"):
            opts.useFloat16 = self.use_float16
        return opts

    def _index_to_gpu(self, index):
        if self.on_gpu:
            return faiss.index_cpu_to_gpu(self._gpu_res, self.device, index, self._gpu_cloner_options())
        return index

    def _index_to_cpu(self, index):
        return faiss.index_gpu_to_cpu(index) if self.on_gpu else index

    def _create_index(self, dimension, force_cpu: bool = False):
        dimension = int(dimension)
        if self.on_gpu and not force_cpu:
            cfg = faiss.GpuIndexFlatConfig()
            cfg.device = self.device
            if hasattr(cfg, "useFloat16"):
                cfg.useFloat16 = self.use_float16
            return faiss.GpuIndexFlatL2(self._gpu_res, dimension, cfg)
        return faiss.IndexFlatL2(dimension)

    @staticmethod
    def _to_float32_contig(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return x if x.flags["C_CONTIGUOUS"] else np.ascontiguousarray(x)

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        x = x.copy()
        faiss.normalize_L2(x)
        return x

    def _prepare_features(self, features: np.ndarray) -> np.ndarray:
        features = self._to_float32_contig(features)
        if self.metric == "cosine":
            features = self._l2_normalize(features)
        return features

    def fit(self, features: np.ndarray) -> None:
        if self.search_index is not None:
            self.reset_index()
        features = self._prepare_features(features)
        self.search_index = self._create_index(features.shape[-1])
        self.search_index.add(features)

    def _postprocess_distances(self, distances: np.ndarray) -> np.ndarray:
        return distances / 2.0 if self.metric == "cosine" else distances

    def run(
        self,
        n_nearest_neighbours: int,
        query_features: np.ndarray,
        index_features: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(query_features, torch.Tensor):
            query_features = query_features.detach().cpu().numpy()
        query_features = self._prepare_features(query_features)

        if index_features is None:
            if self.search_index is None:
                raise RuntimeError("Search index is empty. Call fit() first.")
            distances, indices = self.search_index.search(query_features, int(n_nearest_neighbours))
            return self._postprocess_distances(distances), indices

        if isinstance(index_features, torch.Tensor):
            index_features = index_features.detach().cpu().numpy()
        index_features = self._prepare_features(index_features)
        index = self._create_index(index_features.shape[-1], force_cpu=self.temp_index_on_cpu)
        index.add(index_features)
        distances, indices = index.search(query_features, int(n_nearest_neighbours))
        del index
        return self._postprocess_distances(distances), indices

    def save(self, filename: str) -> None:
        faiss.write_index(self._index_to_cpu(self.search_index), filename)

    def load(self, filename: str) -> None:
        index = faiss.read_index(filename)
        self.search_index = self._index_to_gpu(index) if self.on_gpu else index

    def reset_index(self) -> None:
        if self.search_index is not None:
            self.search_index.reset()
            self.search_index = None
