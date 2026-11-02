import torch
import yaml
import os
import argparse
import csv
import json
from models.patchcore import PatchCore
from utils.dataset import get_data_loader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def visualize_anomaly_map(image_path, anomaly_map, score, label, save_path=None):
    """保存原图、异常热力图和叠加图。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)
    anomaly_map_np = anomaly_map.detach().cpu().numpy()

    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    im1 = axes[1].imshow(anomaly_map_np, cmap='jet')
    axes[1].set_title(f'Anomaly Map (Score: {score:.4f})')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])

    axes[2].imshow(image_np)
    axes[2].imshow(
        anomaly_map_np,
        cmap='jet',
        alpha=0.5,
        extent=(0, image_np.shape[1], image_np.shape[0], 0),
    )
    axes[2].set_title(f'Overlay (Label: {label})')
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved visualization to {save_path}")
    plt.close()

class DataDirDataset:
    """针对data目录结构的自定义数据集"""
    def __init__(self, data_dir, split="test", transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.samples = []
        
        # 加载数据
        self._load_samples()
    
    def _load_samples(self):
        """加载样本"""
        # 测试时使用所有样本
        good_dir = os.path.join(self.data_dir, "good")
        defective_dir = os.path.join(self.data_dir, "defective")
        
        if os.path.exists(good_dir):
            for img_file in os.listdir(good_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(good_dir, img_file)
                    self.samples.append((img_path, 0, "good"))  # 正常样本标签为0
        
        if os.path.exists(defective_dir):
            for img_file in os.listdir(defective_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(defective_dir, img_file)
                    self.samples.append((img_path, 1, "defective"))  # 缺陷样本标签为1
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, sample_type = self.samples[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 图像转换
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long), img_path, sample_type

def select_balanced_indices(dataset, num_samples):
    """限制样本数时兼顾正常和缺陷，避免只抽到目录中的第一类。"""
    if num_samples <= 0 or num_samples >= len(dataset.samples):
        return list(range(len(dataset.samples)))

    good_indices = [i for i, sample in enumerate(dataset.samples) if sample[1] == 0]
    defect_indices = [i for i, sample in enumerate(dataset.samples) if sample[1] == 1]
    if not good_indices or not defect_indices:
        return list(range(num_samples))

    half = max(1, num_samples // 2)
    n_defect = min(len(defect_indices), half)
    n_good = min(len(good_indices), num_samples - n_defect)
    if n_good + n_defect < num_samples:
        n_defect = min(len(defect_indices), num_samples - n_good)
    return good_indices[:n_good] + defect_indices[:n_defect]


def get_data_loader_data_dir(
    data_dir,
    split="test",
    batch_size=1,
    img_size=(256, 256),
    shuffle=False,
    num_workers=0,
    num_samples=0,
):
    """针对data目录结构的数据加载器"""
    dataset = DataDirDataset(data_dir, split=split)

    if 0 < num_samples < len(dataset):
        dataset = torch.utils.data.Subset(
            dataset, select_balanced_indices(dataset, num_samples)
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def evaluate_and_save(results, output_dir, threshold=None):
    """计算图像级指标并保存 JSON、CSV 和汇总指标图。"""
    if not results:
        raise ValueError("没有可评估的推理结果。")

    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    y_true = np.asarray([item["label"] for item in results], dtype=int)
    y_score = np.asarray([item["score"] for item in results], dtype=float)
    has_both_classes = len(np.unique(y_true)) == 2

    normal_scores = y_score[y_true == 0]
    defect_scores = y_score[y_true == 1]
    normal_p95 = float(np.percentile(normal_scores, 95)) if normal_scores.size else None
    normal_p99 = float(np.percentile(normal_scores, 99)) if normal_scores.size else None

    best_threshold = None
    best_f1 = None
    best_precision = None
    best_recall = None
    if has_both_classes:
        unique_scores = np.unique(y_score)
        # Evaluate every possible split under the same "score > threshold" rule.
        thresholds = np.concatenate(
            (
                [np.nextafter(unique_scores[0], -np.inf)],
                (unique_scores[:-1] + unique_scores[1:]) / 2,
            )
        )
        f1_values = []
        precisions = []
        recalls = []
        for candidate in thresholds:
            candidate_pred = (y_score > candidate).astype(int)
            candidate_tn, candidate_fp, candidate_fn, candidate_tp = confusion_matrix(
                y_true, candidate_pred, labels=[0, 1]
            ).ravel()
            candidate_precision = (
                candidate_tp / (candidate_tp + candidate_fp)
                if candidate_tp + candidate_fp
                else 0.0
            )
            candidate_recall = (
                candidate_tp / (candidate_tp + candidate_fn)
                if candidate_tp + candidate_fn
                else 0.0
            )
            candidate_f1 = (
                2 * candidate_precision * candidate_recall
                / (candidate_precision + candidate_recall + 1e-12)
            )
            f1_values.append(candidate_f1)
            precisions.append(candidate_precision)
            recalls.append(candidate_recall)
        f1_values = np.asarray(f1_values)
        best_index = int(np.argmax(f1_values))
        best_threshold = float(thresholds[best_index])
        best_f1 = float(f1_values[best_index])
        best_precision = float(precisions[best_index])
        best_recall = float(recalls[best_index])

    if threshold is None:
        used_threshold = best_threshold
        threshold_source = "f1_optimal_on_evaluation_set"
    else:
        used_threshold = float(threshold)
        threshold_source = "cli_or_function_argument"

    predictions = (
        y_score > used_threshold
        if used_threshold is not None
        else np.zeros_like(y_true, dtype=bool)
    )
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions.astype(int), labels=[0, 1]
    ).ravel()

    accuracy = float((tp + tn) / len(y_true))
    precision_value = float(tp / (tp + fp)) if tp + fp else 0.0
    recall_value = float(tp / (tp + fn)) if tp + fn else 0.0
    f1_value = (
        2 * precision_value * recall_value / (precision_value + recall_value)
        if precision_value + recall_value
        else 0.0
    )

    auroc = float(roc_auc_score(y_true, y_score)) if has_both_classes else None
    average_precision = (
        float(average_precision_score(y_true, y_score))
        if has_both_classes
        else None
    )

    summary = {
        "num_samples": int(len(y_true)),
        "num_good": int((y_true == 0).sum()),
        "num_defective": int((y_true == 1).sum()),
        "auroc": auroc,
        "average_precision": average_precision,
        "threshold": used_threshold,
        "threshold_source": threshold_source,
        "best_f1_threshold": best_threshold,
        "best_f1_on_evaluation_set": best_f1,
        "accuracy": accuracy,
        "precision": precision_value,
        "recall": recall_value,
        "f1": f1_value,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "normal_p95": normal_p95,
        "normal_p99": normal_p99,
        "normal_score_min": float(normal_scores.min()) if normal_scores.size else None,
        "normal_score_max": float(normal_scores.max()) if normal_scores.size else None,
        "defect_score_min": float(defect_scores.min()) if defect_scores.size else None,
        "defect_score_max": float(defect_scores.max()) if defect_scores.size else None,
    }

    with open(os.path.join(metrics_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(metrics_dir, "predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "sample_type",
                "true_label",
                "anomaly_score",
                "predicted_anomaly",
            ],
        )
        writer.writeheader()
        for item, predicted in zip(results, predictions):
            writer.writerow(
                {
                    "image_path": item["image_path"],
                    "sample_type": item["sample_type"],
                    "true_label": item["label"],
                    "anomaly_score": item["score"],
                    "predicted_anomaly": int(predicted),
                }
            )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    bins = max(10, min(40, len(y_score) // 20))
    if normal_scores.size:
        axes[0, 0].hist(
            normal_scores, bins=bins, alpha=0.7, label="good", color="#4878a8"
        )
    if defect_scores.size:
        axes[0, 0].hist(
            defect_scores,
            bins=bins,
            alpha=0.7,
            label="defective",
            color="#c64b4b",
        )
    if used_threshold is not None:
        axes[0, 0].axvline(
            used_threshold,
            color="#222222",
            linestyle="--",
            label=f"threshold={used_threshold:.4f}",
        )
    if normal_p95 is not None:
        axes[0, 0].axvline(
            normal_p95,
            color="#b8782a",
            linestyle=":",
            label=f"good P95={normal_p95:.4f}",
        )
    axes[0, 0].set_title("Anomaly score distribution")
    axes[0, 0].set_xlabel("Image anomaly score")
    axes[0, 0].set_ylabel("Number of images")
    axes[0, 0].legend(frameon=False)

    if has_both_classes:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        axes[0, 1].plot(fpr, tpr, color="#222222", lw=2, label=f"AUC={auroc:.4f}")
        axes[0, 1].plot([0, 1], [0, 1], color="#aaaaaa", linestyle=":")
        axes[0, 1].set_title("Image-level ROC")
        axes[0, 1].set_xlabel("False positive rate")
        axes[0, 1].set_ylabel("True positive rate")
        axes[0, 1].legend(frameon=False)

        curve_precision, curve_recall, _ = precision_recall_curve(y_true, y_score)
        axes[1, 0].plot(
            curve_recall,
            curve_precision,
            color="#222222",
            lw=2,
            label=f"AP={average_precision:.4f}",
        )
        axes[1, 0].set_title("Image-level precision-recall")
        axes[1, 0].set_xlabel("Recall")
        axes[1, 0].set_ylabel("Precision")
        axes[1, 0].legend(frameon=False)
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "ROC requires both good and defective samples",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
        )
        axes[0, 1].set_axis_off()
        axes[1, 0].text(
            0.5,
            0.5,
            "PR curve requires both good and defective samples",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )
        axes[1, 0].set_axis_off()

    cm = np.asarray([[tn, fp], [fn, tp]], dtype=int)
    axes[1, 1].imshow(cm, cmap="Blues")
    axes[1, 1].set_xticks([0, 1], labels=["Pred normal", "Pred anomaly"])
    axes[1, 1].set_yticks([0, 1], labels=["True normal", "True anomaly"])
    axes[1, 1].set_title("Confusion matrix at selected threshold")
    for row in range(2):
        for col in range(2):
            axes[1, 1].text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                color="#111111",
                fontsize=13,
            )

    for axis in axes.ravel():
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("PatchCore directory evaluation", fontsize=15)
    fig.tight_layout()
    plot_path = os.path.join(metrics_dir, "evaluation_metrics.png")
    fig.savefig(plot_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("\n=== Evaluation summary ===")
    print(f"AUROC: {auroc:.4f}" if auroc is not None else "AUROC: N/A")
    print(
        f"Threshold: {used_threshold:.4f} ({threshold_source})"
        if used_threshold is not None
        else "Threshold: N/A"
    )
    print(
        f"Precision: {precision_value:.4f} | Recall: {recall_value:.4f} | "
        f"F1: {f1_value:.4f}"
    )
    print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"Metrics: {metrics_dir}")
    return summary, plot_path

def infer_data_dir(
    model_path,
    data_dir,
    config_path="./configs/config.yaml",
    num_samples=0,
    threshold=None,
    output_dir="./inference_results_data_dir",
    retrieval_backend="torch",
):
    """针对data目录的推理函数"""
    print("=== Starting PatchCore Inference for Data Directory ===")
    
    # 加载配置
    config = load_config(config_path)
    
    # 设置设备
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 初始化模型
    model = PatchCore(
        backbone_name=config['model']['backbone'],
        layers=config['model']['layers'],
        patch_size=config['model']['patch_size'],
        sampling_ratio=config['model']['sampling_ratio'],
        device=device,
        retrieval_backend=retrieval_backend,
    )
    
    # 加载模型权重
    print(f"Loading model from: {model_path}")
    if os.path.exists(model_path):
        model.load_model(model_path)
        print("Model loaded successfully.")
    else:
        print(f"Model file not found: {model_path}")
        return
    
    # 获取测试数据加载器
    print(f"Loading test data from: {data_dir}")
    test_loader = get_data_loader_data_dir(
        data_dir=data_dir,
        split="test",
        batch_size=1,  # 单张图片推理
        img_size=(256, 256),
        shuffle=False,
        num_workers=0,
        num_samples=num_samples,
    )
    
    # 检查测试数据加载器
    print(f"Number of test batches: {len(test_loader)}")
    print(f"Test data loaded with {len(test_loader.dataset)} samples.")
    
    # 创建结果目录
    results_dir = output_dir
    os.makedirs(results_dir, exist_ok=True)
    results = []
    
    # 对测试数据进行推理
    print("Starting inference...")
    model.eval()
    
    with torch.no_grad():
        for i, (images, labels, img_paths, sample_types) in enumerate(test_loader):
            images = images.to(device)
            img_path = img_paths[0]
            sample_type = sample_types[0]
            
            print(f"\nProcessing sample {i+1}/{len(test_loader)}:")
            print(f"  Image: {os.path.basename(img_path)}")
            print(f"  Type: {sample_type}")
            print(f"  Label: {labels.item()}")
            print(f"  Image shape: {images.shape}")
            
            # 进行推理
            anomaly_maps, anomaly_scores = model.predict(images)
            
            print(f"  Anomaly map shape: {anomaly_maps.shape}")
            print(f"  Anomaly score: {anomaly_scores.item():.4f}")
            
            score = float(anomaly_scores.item())

            # 手动阈值可选；未指定时在全部推理完成后用评估集 F1 最优点。
            if threshold is not None and score > threshold:
                anomaly_status = "ANOMALY DETECTED"
            else:
                anomaly_status = "NORMAL" if threshold is not None else "UNCLASSIFIED"
            
            print(f"  Status: {anomaly_status}")
            
            # 可视化结果
            save_path = os.path.join(
                results_dir, f"{sample_type}_{i+1:03d}_score_{score:.4f}.png"
            )
            visualize_anomaly_map(
                img_path,
                anomaly_maps[0],
                score,
                labels.item(),
                save_path=save_path
            )

            results.append(
                {
                    "image_path": img_path,
                    "sample_type": sample_type,
                    "label": int(labels.item()),
                    "score": score,
                }
            )
            
            # 处理指定数量的样本
            if num_samples > 0 and i + 1 >= num_samples:
                break
    
    metrics, plot_path = evaluate_and_save(results, results_dir, threshold=threshold)

    print(f"\nInference completed! Results saved to {results_dir}")
    print(f"Total samples processed: {len(results)}")
    print(f"Metric plot: {plot_path}")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PatchCore directory inference")
    parser.add_argument(
        "--model_path",
        default="./results/patchcore_model_data_dir.pth",
        help="Path to the trained checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        default="./data/chilun/test",
        help="Directory containing good/ and defective/",
    )
    parser.add_argument("--config", default="./configs/config.yaml")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="0 processes all images; otherwise uses a class-balanced subset",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional image-score threshold; omitted means F1-optimal on this evaluation set",
    )
    parser.add_argument(
        "--output_dir",
        default="./inference_results_data_dir",
        help="Directory for visualizations and metrics",
    )
    parser.add_argument(
        "--retrieval_backend",
        choices=["torch", "faiss", "auto"],
        default="torch",
        help="Exact KNN backend for Memory Bank retrieval",
    )
    args = parser.parse_args()

    infer_data_dir(
        args.model_path,
        args.data_dir,
        config_path=args.config,
        num_samples=args.num_samples,
        threshold=args.threshold,
        output_dir=args.output_dir,
        retrieval_backend=args.retrieval_backend,
    )
