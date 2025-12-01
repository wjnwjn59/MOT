"""
Unified fine-tuning script for SimTrack with Classification Head

Supports two modes:
1) BBox-only training (original SimTrack prediction head only)
2) Joint training: BBox + Classification head

- Uses MVTD-style maritime annotations (JSONL) produced by your pipeline.
- Uses SimTrackWithClassification as the base model.
"""

import os
import sys
import json
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from tqdm import tqdm
import argparse

# Add local path (so simtrack_with_classification and lib are found)
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# Import custom model
from simtrack_with_classification import (
    SimTrackWithClassification,
    build_simtrack_with_classification,
    load_pretrained_weights,
    freeze_backbone,
    unfreeze_backbone,
    DummyConfig,
)

# Multi-GPU helper
from lib.train.admin import multigpu


# -----------------------------
# Utils: seeding and boxes
# -----------------------------

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from cxcywh to xyxy format (normalized)."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def compute_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between predicted and target boxes.
    Both inputs are in cxcywh format and normalized [0,1].
    Returns IoU per box: shape (B,)
    """
    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    target_xyxy = box_cxcywh_to_xyxy(target_boxes)

    x1 = torch.max(pred_xyxy[:, 0], target_xyxy[:, 0])
    y1 = torch.max(pred_xyxy[:, 1], target_xyxy[:, 1])
    x2 = torch.min(pred_xyxy[:, 2], target_xyxy[:, 2])
    y2 = torch.min(pred_xyxy[:, 3], target_xyxy[:, 3])

    inter_w = torch.clamp(x2 - x1, min=0)
    inter_h = torch.clamp(y2 - y1, min=0)
    intersection = inter_w * inter_h

    pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
    tgt_area = (target_xyxy[:, 2] - target_xyxy[:, 0]) * (target_xyxy[:, 3] - target_xyxy[:, 1])

    union = pred_area + tgt_area - intersection + 1e-8
    iou = intersection / union
    return iou


def giou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute a simple GIoU loss in cxcywh space.
    Returns scalar loss.
    """
    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    target_xyxy = box_cxcywh_to_xyxy(target_boxes)

    # IoU
    iou = compute_iou(pred_boxes, target_boxes)  # (B,)

    # Enclosing box
    x1_c = torch.min(pred_xyxy[:, 0], target_xyxy[:, 0])
    y1_c = torch.min(pred_xyxy[:, 1], target_xyxy[:, 1])
    x2_c = torch.max(pred_xyxy[:, 2], target_xyxy[:, 2])
    y2_c = torch.max(pred_xyxy[:, 3], target_xyxy[:, 3])

    enclosing_area = (x2_c - x1_c) * (y2_c - y1_c)

    # Pred & target areas
    pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
    tgt_area = (target_xyxy[:, 2] - target_xyxy[:, 0]) * (target_xyxy[:, 3] - target_xyxy[:, 1])

    # Approximate union (using IoU to estimate intersection)
    # Note: This mirrors your existing implementation; not mathematically perfect but consistent.
    union_area = pred_area + tgt_area - iou * pred_area

    giou = iou - (enclosing_area - union_area) / (enclosing_area + 1e-8)
    loss = 1.0 - giou.mean()
    return loss


# -----------------------------
# Dataset
# -----------------------------

class MaritimeMVTDataset(Dataset):
    """
    Dataset for maritime tracking + classification on MVTD-style JSONL annotations.

    Expects each sequence folder under data_dir to contain:
        - template_crop.jpg
        - one *.jsonl file with records like:
          {
            "sequence_name": "...",
            "frame_file": "img0001.jpg",
            "template_bbox": [x1, y1, x2, y2],
            "ground_truth_bbox": [x1, y1, x2, y2],
            "vlm_response": {...},
            "cv_response": {...},
            "dataset_path": "/path/to/MVTD/split"
          }
    """

    def __init__(self, data_dir, transform=None, max_samples_per_seq=None):
        self.data_dir = data_dir
        self.transform = transform
        self.max_samples_per_seq = max_samples_per_seq

        self.samples = []
        self.class_mapping = self._build_class_mapping()
        self._load_annotations()

        print(f"Loaded {len(self.samples)} samples from {len(os.listdir(data_dir))} sequences")
        print(f"Class mapping: {self.class_mapping}")

    def _build_class_mapping(self):
        """Build mapping from class names to indices."""
        vlm_classes = [
            "motion_blur", "illu_change", "variance_appear",
            "partial_visibility", "background_clutter", "occlusion",
        ]
        cv_classes = ["scale_variation", "low_res", "low_contrast"]
        all_classes = vlm_classes + cv_classes + ["normal"]
        return {cls: idx for idx, cls in enumerate(all_classes)}

    def _get_difficulty_class(self, vlm_response, cv_response):
        """
        Determine the difficulty class from VLM and CV responses.

        Priority:
        1. Any VLM flag == 1
        2. Any CV flag == 1
        3. Otherwise "normal"
        """
        # VLM-based flags
        for key, value in vlm_response.items():
            if isinstance(value, dict) and value.get("flag", 0) == 1:
                return key

        # CV-based flags
        for key, value in cv_response.items():
            if value == 1:
                return key

        return "normal"

    def _load_annotations(self):
        seq_dirs = [
            d
            for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ]

        for seq_dir in seq_dirs:
            seq_path = os.path.join(self.data_dir, seq_dir)
            jsonl_files = [p for p in os.listdir(seq_path) if p.endswith(".jsonl")]
            if not jsonl_files:
                continue

            jsonl_file = os.path.join(seq_path, jsonl_files[0])
            template_crop = os.path.join(seq_path, "template_crop.jpg")

            if not os.path.exists(template_crop):
                print(f"Warning: Template crop not found for {seq_dir}")
                continue

            seq_samples = []
            with open(jsonl_file, "r") as f:
                for line_idx, line in enumerate(f):
                    if self.max_samples_per_seq and line_idx >= self.max_samples_per_seq:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)

                        # Class label
                        difficulty_class = self._get_difficulty_class(
                            data["vlm_response"], data["cv_response"]
                        )
                        if difficulty_class not in self.class_mapping:
                            print(f"Unknown class: {difficulty_class}")
                            continue

                        sample = {
                            "sequence_name": data["sequence_name"],
                            "frame_file": data["frame_file"],
                            "template_crop": template_crop,
                            "template_bbox": data["template_bbox"],
                            "ground_truth_bbox": data["ground_truth_bbox"],
                            "dataset_path": data["dataset_path"],
                            "class_label": self.class_mapping[difficulty_class],
                            "class_name": difficulty_class,
                        }
                        seq_samples.append(sample)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_idx} in {jsonl_file}: {e}")
                        continue

            self.samples.extend(seq_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Template image
        template_img = Image.open(sample["template_crop"]).convert("RGB")

        # Search image
        search_img_path = os.path.join(
            "../..",
            sample["dataset_path"],
            sample["sequence_name"],
            sample["frame_file"],
        )
        if not os.path.exists(search_img_path):
            print(f"Warning: Search image not found {search_img_path}, using template")
            search_img = template_img.copy()
        else:
            search_img = Image.open(search_img_path).convert("RGB")

        template_tensor = self._preprocess_image(template_img, size=112)
        search_tensor = self._preprocess_image(search_img, size=224)

        template_bbox = torch.tensor(sample["template_bbox"], dtype=torch.float32)
        gt_bbox = torch.tensor(sample["ground_truth_bbox"], dtype=torch.float32)

        if search_img.size[0] > 1 and search_img.size[1] > 1:
            gt_bbox_norm = self._normalize_bbox(gt_bbox, search_img.size)
            template_bbox_norm = self._normalize_bbox(template_bbox, search_img.size)
        else:
            gt_bbox_norm = torch.tensor([0.5, 0.5, 0.2, 0.2])
            template_bbox_norm = torch.tensor([0.5, 0.5, 0.2, 0.2])

        return {
            "template_img": template_tensor.float(),
            "search_img": search_tensor.float(),
            "template_anno": template_bbox_norm.float(),
            "gt_bbox": gt_bbox_norm.float(),
            "class_label": torch.tensor(sample["class_label"], dtype=torch.long),
            "class_name": sample["class_name"],
            "sequence_name": sample["sequence_name"],
        }

    def _preprocess_image(self, img, size: int):
        """Resize → normalize (ImageNet) → to tensor CHW."""
        img = img.resize((size, size), Image.Resampling.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        arr = (arr - mean) / std

        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float()
        return tensor

    def _normalize_bbox(self, bbox, img_size):
        """Convert xyxy in pixels → normalized cxcywh."""
        w, h = img_size
        x1, y1, x2, y2 = bbox

        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1:
            x2 = x1 + 1
        if y2 <= y1:
            y2 = y1 + 1

        x1_n, x2_n = x1 / w, x2 / w
        y1_n, y2_n = y1 / h, y2 / h

        cx = (x1_n + x2_n) / 2
        cy = (y1_n + y2_n) / 2
        width = x2_n - x1_n
        height = y2_n - y1_n

        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        width = max(0.01, min(1, width))
        height = max(0.01, min(1, height))

        return torch.tensor([cx, cy, width, height], dtype=torch.float32)


def compute_class_weights(dataset: MaritimeMVTDataset, num_classes: int):
    """Compute inverse-frequency class weights for classification loss."""
    class_counts = np.zeros(num_classes)
    for sample in dataset.samples:
        class_counts[sample["class_label"]] += 1

    class_counts = np.maximum(class_counts, 1)
    total_samples = len(dataset.samples)
    weights = total_samples / (num_classes * class_counts)
    weights = weights / np.sum(weights) * num_classes
    return weights


# -----------------------------
# Trainer
# -----------------------------

class MaritimeTrainer:
    """
    Joint trainer for:
      - BBox-only training (train_cls_head = False)
      - Joint BBox + Classification training (train_cls_head = True)
    """

    def __init__(
        self,
        model,
        device,
        save_dir,
        train_cls_head: bool,
        class_weights=None,
        lambda_cls: float = 1.0,
        lambda_l1: float = 5.0,
        lambda_giou: float = 2.0,
    ):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.train_cls_head = train_cls_head
        self.lambda_cls = lambda_cls
        self.lambda_l1 = lambda_l1
        self.lambda_giou = lambda_giou

        if self.train_cls_head and class_weights is not None:
            cw = torch.tensor(class_weights, dtype=torch.float32).to(device)
            print(f"Using class weights: {cw}")
            self.classification_loss = nn.CrossEntropyLoss(weight=cw)
        elif self.train_cls_head:
            self.classification_loss = nn.CrossEntropyLoss()
        else:
            self.classification_loss = None

        self.l1_loss = nn.L1Loss()

        self.best_iou = 0.0
        self.train_losses = []
        self.val_ious = []

    def train_epoch(self, dataloader, optimizer, epoch):
        self.model.train()

        total_loss = 0.0
        total_l1 = 0.0
        total_giou = 0.0
        total_cls = 0.0

        total_iou = 0.0
        total_samples = 0

        total_correct = 0
        total_cls_samples = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            template_img = batch["template_img"].to(self.device)
            search_img = batch["search_img"].to(self.device)
            template_anno = batch["template_anno"].to(self.device)
            gt_bbox = batch["gt_bbox"].to(self.device)
            class_labels = batch["class_label"].to(self.device)

            optimizer.zero_grad()

            try:
                input_data = [template_img, search_img, template_anno]
                backbone_output = self.model.forward_backbone(input_data)

                seq_dict = [backbone_output]
                head_output = self.model.forward_head(
                    seq_dict,
                    run_box_head=True,
                    run_cls_head=self.train_cls_head,
                )

                # --- Box prediction ---
                pred_boxes = head_output["pred_boxes"].squeeze(1)  # (B,4)

                if pred_boxes.size(0) != gt_bbox.size(0):
                    print(f"Batch size mismatch: pred {pred_boxes.shape}, gt {gt_bbox.shape}")
                    continue

                l1 = self.l1_loss(pred_boxes, gt_bbox)
                gl = giou_loss(pred_boxes, gt_bbox)

                # --- Classification (optional) ---
                if self.train_cls_head:
                    cls_logits = head_output["cls_logits"].float()
                    cls_loss = self.classification_loss(cls_logits, class_labels)
                    total_cls += cls_loss.item()

                    pred_cls = torch.argmax(cls_logits, dim=1)
                    total_correct += (pred_cls == class_labels).sum().item()
                    total_cls_samples += class_labels.size(0)
                else:
                    cls_loss = 0.0

                loss = (
                    self.lambda_l1 * l1
                    + self.lambda_giou * gl
                    + (self.lambda_cls * cls_loss if self.train_cls_head else 0.0)
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_l1 += l1.item()
                total_giou += gl.item()

                with torch.no_grad():
                    iou = compute_iou(pred_boxes, gt_bbox)
                    total_iou += iou.mean().item()
                    total_samples += pred_boxes.size(0)

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

            if batch_idx % 10 == 0:
                avg_iou = total_iou / max(total_samples, 1)
                postfix = {
                    "Loss": f"{total_loss/(batch_idx+1):.4f}",
                    "L1": f"{total_l1/(batch_idx+1):.4f}",
                    "GIoU": f"{total_giou/(batch_idx+1):.4f}",
                    "IoU": f"{avg_iou:.3f}",
                }
                if self.train_cls_head and total_cls_samples > 0:
                    acc = total_correct / total_cls_samples * 100
                    postfix["ClsLoss"] = f"{total_cls/(batch_idx+1):.4f}"
                    postfix["Acc"] = f"{acc:.2f}%"
                pbar.set_postfix(postfix)

        avg_loss = total_loss / len(dataloader)
        avg_l1 = total_l1 / len(dataloader)
        avg_giou = total_giou / len(dataloader)
        avg_iou = total_iou / len(dataloader)
        self.train_losses.append(avg_loss)

        print(
            f"Epoch {epoch} - Loss: {avg_loss:.4f}, "
            f"L1: {avg_l1:.4f}, GIoU: {avg_giou:.4f}, IoU: {avg_iou:.3f}"
        )
        if self.train_cls_head and total_cls_samples > 0:
            train_acc = total_correct / total_cls_samples * 100
            print(f"  Classification Accuracy: {train_acc:.2f}%")

        return avg_loss, avg_iou

    def validate(self, dataloader, epoch):
        self.model.eval()

        total_loss = 0.0
        total_l1 = 0.0
        total_giou = 0.0
        total_iou = 0.0
        total_samples = 0

        total_cls_loss = 0.0
        total_correct = 0
        total_cls_samples = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                template_img = batch["template_img"].to(self.device)
                search_img = batch["search_img"].to(self.device)
                template_anno = batch["template_anno"].to(self.device)
                gt_bbox = batch["gt_bbox"].to(self.device)
                class_labels = batch["class_label"].to(self.device)

                try:
                    input_data = [template_img, search_img, template_anno]
                    backbone_output = self.model.forward_backbone(input_data)

                    seq_dict = [backbone_output]
                    head_output = self.model.forward_head(
                        seq_dict,
                        run_box_head=True,
                        run_cls_head=self.train_cls_head,
                    )

                    pred_boxes = head_output["pred_boxes"].squeeze(1)

                    if pred_boxes.size(0) != gt_bbox.size(0):
                        continue

                    l1 = self.l1_loss(pred_boxes, gt_bbox)
                    gl = giou_loss(pred_boxes, gt_bbox)
                    cls_loss_val = 0.0

                    if self.train_cls_head:
                        cls_logits = head_output["cls_logits"].float()
                        cls_loss_val = self.classification_loss(cls_logits, class_labels)
                        total_cls_loss += cls_loss_val.item()

                        pred_cls = torch.argmax(cls_logits, dim=1)
                        total_correct += (pred_cls == class_labels).sum().item()
                        total_cls_samples += class_labels.size(0)

                    loss = (
                        self.lambda_l1 * l1
                        + self.lambda_giou * gl
                        + (self.lambda_cls * cls_loss_val if self.train_cls_head else 0.0)
                    )

                    total_loss += loss.item()
                    total_l1 += l1.item()
                    total_giou += gl.item()

                    iou = compute_iou(pred_boxes, gt_bbox)
                    total_iou += iou.sum().item()
                    total_samples += pred_boxes.size(0)

                except Exception as e:
                    print(f"Error in validation: {e}")
                    continue

        avg_loss = total_loss / len(dataloader)
        avg_l1 = total_l1 / len(dataloader)
        avg_giou = total_giou / len(dataloader)
        avg_iou = total_iou / max(total_samples, 1)
        self.val_ious.append(avg_iou)

        print(
            f"Validation - Loss: {avg_loss:.4f}, "
            f"L1: {avg_l1:.4f}, GIoU: {avg_giou:.4f}, IoU: {avg_iou:.3f}"
        )
        if self.train_cls_head and total_cls_samples > 0:
            val_acc = total_correct / total_cls_samples * 100
            avg_cls_loss = total_cls_loss / len(dataloader)
            print(f"  Classification - Loss: {avg_cls_loss:.4f}, Acc: {val_acc:.2f}%")

        return avg_loss, avg_iou

    def save_checkpoint(self, epoch, optimizer, iou, is_best=False):
        """Save model checkpoint (best judged by IoU)."""
        net = self.model.module if multigpu.is_multi_gpu(self.model) else self.model

        state = {
            "epoch": epoch,
            "net_type": type(net).__name__,
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iou": iou,
            "best_iou": self.best_iou,
            "train_losses": self.train_losses,
            "val_ious": self.val_ious,
        }

        ckpt_path = os.path.join(self.save_dir, f"simtrack_finetune_ep{epoch:04d}.pth.tar")
        torch.save(state, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        if is_best:
            best_path = os.path.join(self.save_dir, "simtrack_finetune_best.pth.tar")
            torch.save(state, best_path)
            print(f"Saved best model: {best_path}")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified fine-tune SimTrack on MVTD")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/train_maritime_env_clf_annts",
        help="Path to maritime MVTD-style annotations (per-sequence folders).",
    )
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default="./SimTrackMod/sim-vit-b-16.pth",
        help="Path to pretrained SimTrack weights.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Root directory to save checkpoints. A per-run subfolder will be created.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size."
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Max samples per sequence (limit frames per sequence).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze backbone + bottleneck during training.",
    )
    parser.add_argument(
        "--train_cls_head",
        action="store_true",
        help="If set, train classification head jointly with bbox. "
             "If not set, train bbox only (cls branch disabled).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=59,
        help="Random seed for everything (default: 59).",
    )

    args = parser.parse_args()

    # Seed
    set_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    print("Loading dataset...")
    full_dataset = MaritimeMVTDataset(
        args.data_dir,
        max_samples_per_seq=args.max_samples,
    )

    num_classes = len(full_dataset.class_mapping)
    print(f"Number of classes: {num_classes}")

    class_weights = compute_class_weights(full_dataset, num_classes)
    print(f"Class weights: {class_weights}")

    # Model
    print("Building model...")
    cfg = DummyConfig()
    model = build_simtrack_with_classification(
        cfg,
        num_classes=num_classes,
        hidden_dim=512,
    )

    print(f"Loading pretrained weights from {args.pretrained_weights}...")
    model = load_pretrained_weights(model, args.pretrained_weights, strict=False)

    # Optionally freeze backbone
    if args.freeze_backbone:
        print("Freezing backbone parameters...")
        model = freeze_backbone(model)
    else:
        print("Training backbone + heads.")

    # Move to device
    model = model.to(device)

    # Split dataset
    total_size = len(full_dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create per-run save folder
    mode_str = "joint" if args.train_cls_head else "bbox"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(
        args.save_dir, f"maritime_simtrackcls_{mode_str}_{timestamp}"
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run save directory: {run_dir}")

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    # Trainer
    trainer = MaritimeTrainer(
        model,
        device,
        run_dir,
        train_cls_head=args.train_cls_head,
        class_weights=class_weights,
        lambda_cls=0.1,
        lambda_l1=5.0,
        lambda_giou=2.0,
    )

    # Training loop
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_iou = trainer.train_epoch(train_loader, optimizer, epoch)
        val_loss, val_iou = trainer.validate(val_loader, epoch)

        scheduler.step()

        is_best = val_iou > trainer.best_iou
        if is_best:
            trainer.best_iou = val_iou

        if epoch % 10 == 0 or is_best:
            trainer.save_checkpoint(epoch, optimizer, val_iou, is_best)

    print(f"Training completed! Best IoU: {trainer.best_iou:.3f}")


if __name__ == "__main__":
    main()
