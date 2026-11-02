"""
Coreset 采样。

- 旧版项目快：本质是 RandomSampler（随机抽 10%），不是真正的 greedy coreset。
- 亚马逊官网 patchcore-inspection 默认：ApproximateGreedyCoresetSampler
  （线性投影到 128 维 + 近似 greedy，无需 N×N 全距离矩阵）。
- Anomalib：KCenterGreedy + SparseRandomProjection(eps=0.9)。

本模块默认使用亚马逊官网的 approx_greedy（速度与论文一致），
并保留 anomalib_kcenter 可选。
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.random_projection import SparseRandomProjection
from tqdm import tqdm


class ApproximateGreedyCoresetSampler:
    """
    对齐 amazon-science/patchcore-inspection 的 ApproximateGreedyCoresetSampler。
    官方 sample_training.sh 默认：sampler -p 0.1 approx_greedy_coreset
    """

    def __init__(
        self,
        percentage: float = 0.1,
        device: str | torch.device | None = None,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
    ):
        if not 0 < percentage <= 1:
            raise ValueError("percentage must be in (0, 1]")
        self.percentage = percentage
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.number_of_starting_points = number_of_starting_points
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _reduce_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] == self.dimension_to_project_features_to:
            return features.to(self.device)
        # 与官网一致：随机线性投影到 128 维（不训练）
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        ).to(self.device)
        with torch.no_grad():
            return mapper(features.to(self.device))

    @staticmethod
    def _pairwise_distances(matrix_a: torch.Tensor, matrix_b: torch.Tensor) -> torch.Tensor:
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)
        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def run(self, features: torch.Tensor, progress_cb=None) -> tuple[np.ndarray, torch.Tensor]:
        """
        Returns:
            indices: np.ndarray
            sampled_features: 原始特征子集（CPU）
        """
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        features = features.detach().float()
        n = features.shape[0]
        if self.percentage >= 1.0:
            idx = np.arange(n)
            return idx, features

        reduced = self._reduce_features(features)
        n_start = int(np.clip(self.number_of_starting_points, 1, n))
        start_points = np.random.choice(n, n_start, replace=False).tolist()

        approx_dist = self._pairwise_distances(reduced, reduced[start_points])
        anchor = torch.mean(approx_dist, dim=-1, keepdim=True)

        num_samples = max(1, int(n * self.percentage))
        selected: list[int] = []

        with torch.no_grad():
            iterator = range(num_samples)
            if progress_cb is None:
                iterator = tqdm(iterator, desc="Subsampling(approx_greedy)", leave=False)
            for step in iterator:
                select_idx = int(torch.argmax(anchor).item())
                selected.append(select_idx)
                d_new = self._pairwise_distances(
                    reduced, reduced[select_idx : select_idx + 1]
                )
                anchor = torch.minimum(anchor, d_new)
                # 避免重复选中
                anchor[select_idx] = -1.0
                if progress_cb is not None and (
                    step % max(1, num_samples // 100) == 0 or step + 1 == num_samples
                ):
                    progress_cb((step + 1) / num_samples)

        indices = np.array(selected, dtype=np.int64)
        return indices, features[indices].cpu()


class KCenterGreedy:
    """Anomalib 风格 KCenterGreedy + SparseRandomProjection(eps=0.9)。"""

    def __init__(self, embedding: torch.Tensor, sampling_ratio: float = 0.1):
        n = embedding.shape[0]
        coreset_size = int(n * sampling_ratio)
        if coreset_size <= 0:
            raise ValueError(f"invalid coreset_size={coreset_size}")
        self.embedding = embedding
        self.coreset_size = min(coreset_size, n)
        self.n_observations = n
        self.features = None
        self.min_distances = None

    def _project(self, device: torch.device) -> torch.Tensor:
        x_np = self.embedding.detach().float().cpu().numpy()
        try:
            projector = SparseRandomProjection(eps=0.9, random_state=42)
            x_proj = projector.fit_transform(x_np)
        except ValueError:
            dim = min(128, x_np.shape[0], x_np.shape[1])
            projector = SparseRandomProjection(n_components=dim, random_state=42)
            x_proj = projector.fit_transform(x_np)
        return torch.from_numpy(x_proj).float().to(device)

    def select_coreset_idxs(self, device=None, progress_cb=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        self.features = self._project(device)
        self.min_distances = None

        idx = torch.randint(high=self.n_observations, size=(1,), device=device).squeeze()
        selected = [idx]
        iterator = range(self.coreset_size - 1)
        if progress_cb is None:
            iterator = tqdm(iterator, desc="Selecting Coreset Indices", leave=False)

        for step in iterator:
            center = self.features[idx].squeeze()
            distances = torch.linalg.norm(self.features - center, ord=2, dim=1, keepdim=True)
            self.min_distances = (
                distances
                if self.min_distances is None
                else torch.minimum(self.min_distances, distances)
            )
            _, idx = torch.max(self.min_distances.squeeze(1), dim=0)
            self.min_distances.scatter_(0, idx.view(1, 1), 0.0)
            selected.append(idx)
            if progress_cb is not None and (
                step % max(1, self.coreset_size // 100) == 0
                or step + 1 == self.coreset_size - 1
            ):
                progress_cb((step + 1) / max(1, self.coreset_size - 1))
        return torch.stack(selected).detach().cpu().tolist()


def coreset_sampling(
    features: torch.Tensor,
    sampling_ratio: float = 0.1,
    device: str | torch.device | None = None,
    progress_cb=None,
    method: str = "approx_greedy",
):
    """
    Args:
        method:
          - approx_greedy: 亚马逊官网默认（推荐，快）
          - anomalib_kcenter: Anomalib KCenterGreedy
          - random: 旧项目方法（最快，非官方）
    """
    if method == "random":
        n = features.shape[0]
        m = max(1, int(n * sampling_ratio))
        indices = torch.randperm(n)[:m].numpy()
        return indices, features[indices].detach().cpu()

    if method == "anomalib_kcenter":
        sampler = KCenterGreedy(features.detach(), sampling_ratio=sampling_ratio)
        indices = sampler.select_coreset_idxs(device=device, progress_cb=progress_cb)
        return indices, features[indices].detach().cpu()

    # 默认：官网 approx_greedy
    sampler = ApproximateGreedyCoresetSampler(
        percentage=sampling_ratio,
        device=device,
    )
    return sampler.run(features, progress_cb=progress_cb)
