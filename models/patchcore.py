"""
PatchCore 双模式实现。

- official（官网）:
    AvgPool2d(3,1,1) + 插值拼接 + approx_greedy_coreset + 重加权打分(num_neighbors=9)
- legacy（旧版工程）:
    无 AvgPool + 截断对齐拼接 + 随机 Coreset + 最近邻最大距离打分
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm

from .sampling import coreset_sampling

VARIANT_OFFICIAL = "official"
VARIANT_LEGACY = "legacy"


class PatchCore(nn.Module):
    def __init__(
        self,
        backbone_name="wide_resnet50_2",
        layers=None,
        patch_size=3,
        sampling_ratio=0.1,
        num_neighbors=9,
        device="cuda",
        variant=VARIANT_OFFICIAL,
        coreset_method=None,
        retrieval_backend="torch",
        faiss_gpu=True,
    ):
        super().__init__()
        if layers is None:
            layers = ["layer2", "layer3"]

        self.variant = variant if variant in (VARIANT_OFFICIAL, VARIANT_LEGACY) else VARIANT_OFFICIAL
        self.backbone_name = backbone_name
        self.layers = list(layers)
        self.patch_size = patch_size
        self.sampling_ratio = sampling_ratio
        self.num_neighbors = 1 if self.variant == VARIANT_LEGACY else num_neighbors
        self.device = self._resolve_device(device)
        self.requested_retrieval_backend = retrieval_backend
        self.retrieval_backend = retrieval_backend
        self.faiss_gpu = faiss_gpu
        self._faiss_index = None
        self._faiss_index_cpu = None
        self._faiss_resources = None

        # coreset: 官网 approx_greedy；旧版 random
        if coreset_method is None:
            self.coreset_method = "random" if self.variant == VARIANT_LEGACY else "approx_greedy"
        else:
            self.coreset_method = coreset_method

        self.backbone = self._get_backbone()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 官网才启用邻域 AvgPool；旧版用 Identity
        if self.variant == VARIANT_OFFICIAL:
            self.feature_pooler = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        else:
            self.feature_pooler = nn.Identity()

        self.feature_extractor = self._register_hooks()
        self.memory_bank = None

    @staticmethod
    def _resolve_device(device):
        requested = str(device)
        if requested.startswith("cuda") and not torch.cuda.is_available():
            print(f"Warning: device '{requested}' unavailable, fallback to cpu")
            return "cpu"
        return requested

    def _get_backbone(self):
        weights = "DEFAULT"
        builders = {
            "wide_resnet50_2": models.wide_resnet50_2,
            "resnet18": models.resnet18,
            "resnet50": models.resnet50,
        }
        if self.backbone_name not in builders:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        builder = builders[self.backbone_name]
        try:
            backbone = builder(weights=weights)
        except TypeError:
            backbone = builder(pretrained=True)
        return backbone.to(self.device)

    def _register_hooks(self):
        self.features = {}

        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output

            return hook

        for name, layer in self.backbone.named_modules():
            if name in self.layers:
                layer.register_forward_hook(get_activation(name))
        return self.features

    def forward(self, x):
        self.backbone(x)
        return self.feature_extractor

    def _embed(self, images):
        if self.variant == VARIANT_LEGACY:
            return self._embed_legacy(images)
        return self._embed_official(images)

    def _embed_official(self, images):
        """官网: AvgPool + 插值到 layers[0] 分辨率后拼接。"""
        _ = self.forward(images)
        pooled = {}
        for layer_name in self.layers:
            if layer_name not in self.feature_extractor:
                raise ValueError(
                    f"Layer '{layer_name}' not found. "
                    f"Available: {list(self.feature_extractor.keys())}"
                )
            pooled[layer_name] = self.feature_pooler(self.feature_extractor[layer_name])

        embeddings = pooled[self.layers[0]]
        for layer_name in self.layers[1:]:
            layer_embedding = F.interpolate(
                pooled[layer_name],
                size=embeddings.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            embeddings = torch.cat((embeddings, layer_embedding), dim=1)

        b, c, h, w = embeddings.shape
        combined = embeddings.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return combined, (h, w)

    def _embed_legacy(self, images):
        """旧版: 无 AvgPool，多层特征截断到最短序列后拼接。"""
        _ = self.forward(images)
        layer_features = []
        for layer_name in self.layers:
            if layer_name not in self.feature_extractor:
                raise ValueError(
                    f"Layer '{layer_name}' not found. "
                    f"Available: {list(self.feature_extractor.keys())}"
                )
            feature = self.feature_extractor[layer_name]
            b, c, h, w = feature.shape
            layer_features.append(feature.view(b, c, h * w).permute(0, 2, 1))

        min_seq_len = min(f.shape[1] for f in layer_features)
        adjusted = [f[:, :min_seq_len, :] for f in layer_features]
        combined = torch.cat(adjusted, dim=-1)
        map_size = int(min_seq_len ** 0.5)
        if map_size * map_size != min_seq_len:
            # 无法完美开方时仍返回序列长度提示用的近似边长
            map_size = map_size
        return combined, (map_size, map_size)

    def fit(self, dataloader):
        all_features = []
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Extract features"):
                images = images.to(self.device)
                combined, _ = self._embed(images)
                all_features.append(combined.reshape(-1, combined.shape[-1]).cpu())

        if not all_features:
            raise ValueError("No features were extracted during training.")

        all_features = torch.cat(all_features, dim=0)
        self.memory_bank = self._coreset_sampling(all_features)
        self._build_retrieval_index()

    def fit_with_progress(self, dataloader, progress_bar, status_text, total_batches):
        all_features = []
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(dataloader):
                images = images.to(self.device)
                progress = (batch_idx + 1) / max(total_batches, 1) * 0.7
                progress_bar.progress(min(progress, 0.7))
                status_text.text(
                    f"[{self.variant}] 提取特征... ({batch_idx + 1}/{total_batches}) | 设备: {self.device}"
                )
                combined, _ = self._embed(images)
                all_features.append(combined.reshape(-1, combined.shape[-1]).cpu())

        if not all_features:
            raise ValueError("No features were extracted during training.")

        all_features = torch.cat(all_features, dim=0)
        status_text.text(
            f"[{self.variant}] Coreset({self.coreset_method}) "
            f"N={all_features.shape[0]}, ratio={self.sampling_ratio}"
        )
        progress_bar.progress(0.72)

        def _cb(p):
            progress_bar.progress(0.72 + 0.25 * float(p))
            status_text.text(f"[{self.variant}] Coreset 进度: {p * 100:.1f}%")

        self.memory_bank = self._coreset_sampling(all_features, progress_cb=_cb)
        self._build_retrieval_index()
        progress_bar.progress(0.98)
        status_text.text("Coreset 完成，正在保存...")

    def _coreset_sampling(self, features, progress_cb=None):
        coreset_device = self.device if str(self.device).startswith("cuda") else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        _, sampled = coreset_sampling(
            features,
            sampling_ratio=self.sampling_ratio,
            device=coreset_device,
            progress_cb=progress_cb,
            method=self.coreset_method,
        )
        return sampled.to(self.device)

    @staticmethod
    def _euclidean_dist(x, y):
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)
        y_norm = y.pow(2).sum(dim=-1, keepdim=True)
        res = torch.matmul(x, y.transpose(-2, -1))
        res = res.mul(-2).add(x_norm).add(y_norm.transpose(-2, -1))
        return res.clamp_min_(0).sqrt_()

    def _build_retrieval_index(self):
        self._faiss_index = None
        self._faiss_index_cpu = None
        self._faiss_resources = None
        requested = str(self.requested_retrieval_backend).lower()
        if requested not in {"torch", "faiss", "auto"}:
            raise ValueError("retrieval_backend must be one of: torch, faiss, auto")
        if requested == "torch" or self.memory_bank is None:
            self.retrieval_backend = "torch"
            return
        try:
            import faiss
        except ImportError:
            if requested == "faiss":
                raise ImportError(
                    "FAISS retrieval requested but faiss is not installed. "
                    "Install faiss-cpu or use --retrieval_backend torch."
                )
            print("FAISS is unavailable; retrieval backend falls back to torch.")
            self.retrieval_backend = "torch"
            return

        bank = self.memory_bank.detach().float().cpu().contiguous().numpy()
        index = faiss.IndexFlatL2(bank.shape[1])
        index.add(bank)
        self._faiss_index_cpu = index
        self._faiss_index = index

        use_gpu = (
            self.faiss_gpu
            and str(self.device).startswith("cuda")
            and hasattr(faiss, "index_cpu_to_gpu")
            and hasattr(faiss, "StandardGpuResources")
            and hasattr(faiss, "get_num_gpus")
            and faiss.get_num_gpus() > 0
        )
        if use_gpu:
            self._faiss_resources = faiss.StandardGpuResources()
            self._faiss_index = faiss.index_cpu_to_gpu(
                self._faiss_resources, 0, index
            )
        self.retrieval_backend = "faiss"

    def set_retrieval_backend(self, retrieval_backend="torch", faiss_gpu=True):
        """Switch retrieval backend without changing the Memory Bank."""
        self.requested_retrieval_backend = retrieval_backend
        self.faiss_gpu = faiss_gpu
        self._build_retrieval_index()
        return self.retrieval_backend

    def _nearest_neighbors(self, embedding, n_neighbors):
        k = min(int(n_neighbors), self.memory_bank.shape[0])
        if self.retrieval_backend == "faiss":
            if self._faiss_index is None:
                self._build_retrieval_index()
            query = embedding.detach().float()
            gpu_index = (
                self._faiss_index is not self._faiss_index_cpu
                and str(self.device).startswith("cuda")
            )
            if not gpu_index:
                query = query.cpu()
            query = query.contiguous().numpy()
            distances, indices = self._faiss_index.search(query, k)
            scores = torch.from_numpy(distances).to(embedding.device).clamp_min_(0).sqrt_()
            locations = torch.from_numpy(indices).long().to(self.memory_bank.device)
            return scores, locations

        distances = self._euclidean_dist(embedding, self.memory_bank)
        if k == 1:
            return distances.min(1)
        return distances.topk(k=k, largest=False, dim=1)

    def _compute_anomaly_score(self, patch_scores, locations, embedding):
        if self.variant == VARIANT_LEGACY or self.num_neighbors == 1:
            return patch_scores.amax(1)

        batch_size, num_patches = patch_scores.shape
        max_patches = torch.argmax(patch_scores, dim=1)
        max_patches_features = embedding.reshape(batch_size, num_patches, -1)[
            torch.arange(batch_size, device=embedding.device), max_patches
        ]
        score = patch_scores[torch.arange(batch_size, device=embedding.device), max_patches]
        nn_index = locations[torch.arange(batch_size, device=embedding.device), max_patches]
        nn_sample = self.memory_bank[nn_index, :]

        k = min(self.num_neighbors, self.memory_bank.shape[0])
        _, support_samples = self._nearest_neighbors(nn_sample, n_neighbors=k)
        distances = self._euclidean_dist(
            max_patches_features.unsqueeze(1),
            self.memory_bank[support_samples],
        )
        weights = (1 - F.softmax(distances.squeeze(1), dim=1))[..., 0]
        return weights * score

    def predict(self, image):
        if self.memory_bank is None:
            raise ValueError("Model not trained yet. Call fit() first.")

        with torch.no_grad():
            image = image.to(self.device)
            combined, (map_h, map_w) = self._embed(image)
            batch_size = combined.shape[0]
            embedding = combined.reshape(-1, combined.shape[-1])

            patch_scores, locations = self._nearest_neighbors(embedding, n_neighbors=1)
            patch_scores_b = patch_scores.reshape(batch_size, -1)
            locations_b = locations.reshape(batch_size, -1)

            image_scores = self._compute_anomaly_score(patch_scores_b, locations_b, embedding)

            # legacy 截断后可能无法完美 reshape，做兼容
            n_patches = patch_scores_b.shape[1]
            if map_h * map_w != n_patches:
                side = int(n_patches ** 0.5)
                if side * side == n_patches:
                    map_h = map_w = side
                else:
                    anomaly_maps = patch_scores_b
                    return anomaly_maps, image_scores

            anomaly_maps = patch_scores_b.reshape(batch_size, map_h, map_w)
            return anomaly_maps, image_scores

    def save_model(self, path):
        torch.save(
            {
                "memory_bank": self.memory_bank,
                "backbone_name": self.backbone_name,
                "layers": self.layers,
                "patch_size": self.patch_size,
                "sampling_ratio": self.sampling_ratio,
                "num_neighbors": self.num_neighbors,
                "variant": self.variant,
                "coreset_method": self.coreset_method,
            },
            path,
        )

    def load_model(self, path):
        self.device = self._resolve_device(self.device)
        checkpoint = torch.load(path, map_location=self.device)
        self.memory_bank = checkpoint["memory_bank"]
        if self.memory_bank is not None:
            self.memory_bank = self.memory_bank.to(self.device)
        self.backbone_name = checkpoint["backbone_name"]
        self.layers = checkpoint["layers"]
        self.patch_size = checkpoint["patch_size"]
        self.sampling_ratio = checkpoint["sampling_ratio"]
        saved_neighbors = checkpoint.get("num_neighbors")
        # 无 variant 时默认按官网 official 加载（对齐 Anomalib/亚马逊默认）
        self.variant = checkpoint.get("variant", VARIANT_OFFICIAL)
        if self.variant not in (VARIANT_OFFICIAL, VARIANT_LEGACY):
            self.variant = VARIANT_OFFICIAL
        self.coreset_method = checkpoint.get(
            "coreset_method",
            "random" if self.variant == VARIANT_LEGACY else "approx_greedy",
        )
        if self.variant == VARIANT_OFFICIAL:
            self.feature_pooler = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
            self.num_neighbors = int(saved_neighbors if saved_neighbors is not None else 9)
        else:
            self.feature_pooler = nn.Identity()
            self.num_neighbors = 1
        self.backbone = self._get_backbone()
        self.feature_extractor = self._register_hooks()
        self._build_retrieval_index()
