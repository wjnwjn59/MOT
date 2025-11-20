"""
Fine-tuning script for SimTrack with Prediction Head Only
Trains only the prediction head while using CLS token features for fusion
Classification head is not trained but used for feature extraction
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import argparse
from datetime import datetime
import glob

# Add lib path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# Import our custom model
from simtrack_with_classification import (
    SimTrackWithClassification, 
    build_simtrack_with_classification,
    load_pretrained_weights,
    freeze_backbone,
    unfreeze_backbone,
    DummyConfig
)

# Import utilities from lib
from lib.utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh
from lib.train.admin import multigpu


class MaritimeTrackingDataset(Dataset):
    """Dataset for maritime tracking with prediction head training only"""
    
    def __init__(self, data_dir, transform=None, max_samples_per_seq=None):
        self.data_dir = data_dir
        self.transform = transform
        self.max_samples_per_seq = max_samples_per_seq
        
        # Load all annotation files
        self.samples = []
        self._load_annotations()
        
        print(f"Loaded {len(self.samples)} samples from {len(os.listdir(data_dir))} sequences")
        
    def _load_annotations(self):
        """Load all annotation files"""
        seq_dirs = [d for d in os.listdir(self.data_dir) 
                   if os.path.isdir(os.path.join(self.data_dir, d))]
        
        for seq_dir in seq_dirs:
            seq_path = os.path.join(self.data_dir, seq_dir)
            jsonl_files = glob.glob(os.path.join(seq_path, "*.jsonl"))
            
            if not jsonl_files:
                continue
                
            jsonl_file = jsonl_files[0]  # Take first jsonl file
            template_crop = os.path.join(seq_path, "template_crop.jpg")
            
            if not os.path.exists(template_crop):
                print(f"Warning: Template crop not found for {seq_dir}")
                continue
            
            # Load jsonl data
            seq_samples = []
            with open(jsonl_file, 'r') as f:
                for line_idx, line in enumerate(f):
                    if self.max_samples_per_seq and line_idx >= self.max_samples_per_seq:
                        break
                        
                    try:
                        data = json.loads(line.strip())
                        
                        # Construct sample
                        sample = {
                            'sequence_name': data['sequence_name'],
                            'frame_file': data['frame_file'], 
                            'template_crop': template_crop,
                            'template_bbox': data['template_bbox'],
                            'ground_truth_bbox': data['ground_truth_bbox'],
                            'dataset_path': data['dataset_path']
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
        
        # Load template image
        template_img = Image.open(sample['template_crop']).convert('RGB')
        
        # Load search image
        search_img_path = os.path.join(
            sample['dataset_path'], 
            sample['sequence_name'], 
            sample['frame_file']
        )
        
        if not os.path.exists(search_img_path):
            # Fallback: use template as search (for testing)
            print(f"Warning: Search image not found {search_img_path}, using template")
            search_img = template_img.copy()
        else:
            search_img = Image.open(search_img_path).convert('RGB')
        
        # Convert to tensor and normalize
        template_tensor = self._preprocess_image(template_img, size=112)
        search_tensor = self._preprocess_image(search_img, size=224)
        
        # Process bbox (convert to normalized xywh format)
        template_bbox = torch.tensor(sample['template_bbox'], dtype=torch.float32)
        gt_bbox = torch.tensor(sample['ground_truth_bbox'], dtype=torch.float32)
        
        # Normalize bboxes (assuming they're in pixel coordinates)
        if search_img.size[0] > 1 and search_img.size[1] > 1:  # Valid image
            gt_bbox_norm = self._normalize_bbox(gt_bbox, search_img.size)
            template_bbox_norm = self._normalize_bbox(template_bbox, search_img.size)
        else:
            gt_bbox_norm = torch.tensor([0.5, 0.5, 0.2, 0.2])  # Default
            template_bbox_norm = torch.tensor([0.5, 0.5, 0.2, 0.2])
        
        return {
            'template_img': template_tensor.float(),
            'search_img': search_tensor.float(), 
            'template_anno': template_bbox_norm.float(),
            'gt_bbox': gt_bbox_norm.float(),
            'sequence_name': sample['sequence_name']
        }
    
    def _preprocess_image(self, img, size):
        """Preprocess image to tensor"""
        # Resize
        img = img.resize((size, size), Image.Resampling.BILINEAR)
        
        # Convert to tensor and normalize
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        # HWC to CHW
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
        
        return img_tensor
    
    def _normalize_bbox(self, bbox, img_size):
        """Normalize bbox to [0, 1] and convert to cxcywh format"""
        w, h = img_size
        x1, y1, x2, y2 = bbox
        
        # Clamp to image boundaries first
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Ensure x2 > x1 and y2 > y1
        if x2 <= x1:
            x2 = x1 + 1
        if y2 <= y1:
            y2 = y1 + 1
        
        # Normalize coordinates
        x1, x2 = x1 / w, x2 / w  
        y1, y2 = y1 / h, y2 / h
        
        # Convert to center format
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # Clamp to [0, 1] range
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        width = max(0.01, min(1, width))  # Minimum width
        height = max(0.01, min(1, height))  # Minimum height
        
        return torch.tensor([cx, cy, width, height], dtype=torch.float32)


def compute_iou(pred_boxes, target_boxes):
    """
    Compute IoU between predicted and target boxes
    Both inputs are in cxcywh format and normalized [0,1]
    """
    # Convert to xyxy format
    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    target_xyxy = box_cxcywh_to_xyxy(target_boxes)
    
    # Compute intersection
    x1 = torch.max(pred_xyxy[:, 0], target_xyxy[:, 0])
    y1 = torch.max(pred_xyxy[:, 1], target_xyxy[:, 1])
    x2 = torch.min(pred_xyxy[:, 2], target_xyxy[:, 2])
    y2 = torch.min(pred_xyxy[:, 3], target_xyxy[:, 3])
    
    # Compute intersection area
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Compute areas
    pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
    target_area = (target_xyxy[:, 2] - target_xyxy[:, 0]) * (target_xyxy[:, 3] - target_xyxy[:, 1])
    
    # Compute union
    union = pred_area + target_area - intersection
    
    # Compute IoU
    iou = intersection / (union + 1e-8)
    
    return iou


def box_cxcywh_to_xyxy(boxes):
    """Convert boxes from cxcywh to xyxy format"""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def freeze_classification_head(model):
    """
    Freeze classification head parameters but keep them for feature extraction
    """
    frozen_params = 0
    
    # Freeze classification head components
    for name, param in model.cls_projection.named_parameters():
        param.requires_grad = False
        frozen_params += param.numel()
        print(f"Frozen: cls_projection.{name}")
    
    for name, param in model.classifier.named_parameters():
        param.requires_grad = False
        frozen_params += param.numel()
        print(f"Frozen: classifier.{name}")
    
    # Keep fusion layer trainable for adaptation
    print("Keeping fusion_layer trainable for feature adaptation")
    
    print(f"Frozen {frozen_params:,} classification head parameters")
    return model


class PredictionTrainer:
    """Trainer for prediction head with CLS feature fusion"""
    
    def __init__(self, model, device, save_dir):
        self.model = model
        self.device = device 
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.giou_loss = self._giou_loss
        
        # Metrics
        self.best_iou = 0.0
        self.train_losses = []
        self.val_ious = []
        
    def _giou_loss(self, pred_boxes, target_boxes):
        """Compute GIoU loss"""
        # Convert to xyxy format
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        target_xyxy = box_cxcywh_to_xyxy(target_boxes)
        
        # Compute IoU
        iou = compute_iou(pred_boxes, target_boxes)
        
        # Compute enclosing box
        x1_c = torch.min(pred_xyxy[:, 0], target_xyxy[:, 0])
        y1_c = torch.min(pred_xyxy[:, 1], target_xyxy[:, 1])
        x2_c = torch.max(pred_xyxy[:, 2], target_xyxy[:, 2])
        y2_c = torch.max(pred_xyxy[:, 3], target_xyxy[:, 3])
        
        # Enclosing area
        enclosing_area = (x2_c - x1_c) * (y2_c - y1_c)
        
        # Union area
        pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
        target_area = (target_xyxy[:, 2] - target_xyxy[:, 0]) * (target_xyxy[:, 3] - target_xyxy[:, 1])
        union_area = pred_area + target_area - iou * pred_area  # Using IoU to get intersection
        
        # GIoU
        giou = iou - (enclosing_area - union_area) / (enclosing_area + 1e-8)
        
        # GIoU loss
        giou_loss = 1 - giou.mean()
        
        return giou_loss
        
    def train_epoch(self, dataloader, optimizer, epoch):
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_l1_loss = 0.0 
        total_giou_loss = 0.0
        total_iou = 0.0
        total_samples = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            template_img = batch['template_img'].to(self.device)
            search_img = batch['search_img'].to(self.device)  
            template_anno = batch['template_anno'].to(self.device)
            gt_bbox = batch['gt_bbox'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            try:
                # Backbone forward
                input_data = [template_img, search_img, template_anno]
                backbone_output = self.model.forward_backbone(input_data)
                
                # Head forward (both classification and prediction)
                # Classification head is used for feature extraction only
                seq_dict = [backbone_output]
                head_output = self.model.forward_head(
                    seq_dict, run_box_head=True, run_cls_head=True
                )
                
                # Get prediction boxes
                pred_boxes = head_output['pred_boxes'].squeeze(1)  # Remove query dimension
                
                # Ensure same batch size
                if pred_boxes.size(0) != gt_bbox.size(0):
                    print(f"Batch size mismatch: pred {pred_boxes.shape}, gt {gt_bbox.shape}")
                    continue
                
                # Compute losses
                l1_loss = self.l1_loss(pred_boxes, gt_bbox)
                giou_loss = self.giou_loss(pred_boxes, gt_bbox)
                
                # Combined loss
                loss = l1_loss + giou_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                total_l1_loss += l1_loss.item()
                total_giou_loss += giou_loss.item()
                
                # IoU computation
                with torch.no_grad():
                    iou = compute_iou(pred_boxes, gt_bbox)
                    total_iou += iou.mean().item()
                    total_samples += pred_boxes.size(0)
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
                
            # Update progress bar
            if batch_idx % 10 == 0:
                current_iou = total_iou / max(batch_idx + 1, 1)
                pbar.set_postfix({
                    'Loss': f'{total_loss/(batch_idx+1):.4f}',
                    'L1': f'{total_l1_loss/(batch_idx+1):.4f}',
                    'GIoU': f'{total_giou_loss/(batch_idx+1):.4f}',
                    'IoU': f'{current_iou:.3f}'
                })
        
        # Epoch statistics
        avg_loss = total_loss / len(dataloader)
        avg_l1_loss = total_l1_loss / len(dataloader)
        avg_giou_loss = total_giou_loss / len(dataloader)
        avg_iou = total_iou / len(dataloader)
        
        self.train_losses.append(avg_loss)
        
        print(f'Epoch {epoch} - Loss: {avg_loss:.4f}, L1: {avg_l1_loss:.4f}, GIoU: {avg_giou_loss:.4f}, IoU: {avg_iou:.3f}')
        
        return avg_loss, avg_iou
    
    def validate(self, dataloader, epoch):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        total_l1_loss = 0.0
        total_giou_loss = 0.0
        total_iou = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                # Move to device
                template_img = batch['template_img'].to(self.device)
                search_img = batch['search_img'].to(self.device)
                template_anno = batch['template_anno'].to(self.device) 
                gt_bbox = batch['gt_bbox'].to(self.device)
                
                try:
                    # Forward pass
                    input_data = [template_img, search_img, template_anno]
                    backbone_output = self.model.forward_backbone(input_data)
                    
                    seq_dict = [backbone_output]
                    head_output = self.model.forward_head(
                        seq_dict, run_box_head=True, run_cls_head=True
                    )
                    
                    pred_boxes = head_output['pred_boxes'].squeeze(1)
                    
                    # Ensure same batch size
                    if pred_boxes.size(0) != gt_bbox.size(0):
                        continue
                    
                    # Compute losses
                    l1_loss = self.l1_loss(pred_boxes, gt_bbox)
                    giou_loss = self.giou_loss(pred_boxes, gt_bbox)
                    loss = l1_loss + giou_loss
                    
                    total_loss += loss.item()
                    total_l1_loss += l1_loss.item()
                    total_giou_loss += giou_loss.item()
                    
                    # IoU computation
                    iou = compute_iou(pred_boxes, gt_bbox)
                    total_iou += iou.sum().item()
                    total_samples += pred_boxes.size(0)
                            
                except Exception as e:
                    print(f"Error in validation: {e}")
                    continue
        
        # Statistics
        avg_loss = total_loss / len(dataloader)
        avg_l1_loss = total_l1_loss / len(dataloader)
        avg_giou_loss = total_giou_loss / len(dataloader)
        avg_iou = total_iou / max(total_samples, 1)
        
        self.val_ious.append(avg_iou)
        
        print(f'Validation - Loss: {avg_loss:.4f}, L1: {avg_l1_loss:.4f}, GIoU: {avg_giou_loss:.4f}, IoU: {avg_iou:.3f}')
        
        return avg_loss, avg_iou
    
    def save_checkpoint(self, epoch, optimizer, iou, is_best=False):
        """Save model checkpoint"""
        net = self.model.module if multigpu.is_multi_gpu(self.model) else self.model
        
        state = {
            'epoch': epoch,
            'net_type': type(net).__name__,
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iou': iou,
            'best_iou': self.best_iou,
            'train_losses': self.train_losses,
            'val_ious': self.val_ious
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'simtrack_prediction_ep{epoch:04d}.pth.tar')
        torch.save(state, checkpoint_path)
        print(f'Saved checkpoint: {checkpoint_path}')
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'simtrack_prediction_best.pth.tar')
            torch.save(state, best_path)
            print(f'Saved best model: {best_path}')


def main():
    parser = argparse.ArgumentParser(description='Fine-tune SimTrack Prediction Head with CLS Feature Fusion')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/thinhnp/MOT/data/train_maritime_env_clf_annts',
                       help='Path to maritime tracking data')
    parser.add_argument('--pretrained_weights', type=str,
                       default='/home/thinhnp/MOT/models/SimTrackMod/sim-vit-b-16.pth',
                       help='Path to pretrained weights')
    parser.add_argument('--classification_weights', type=str, default=None,
                       help='Path to pre-trained classification weights (optional)')
    parser.add_argument('--save_dir', type=str,
                       default='/home/thinhnp/MOT/models/SimTrackMod/checkpoints/prediction_only',
                       help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--max_samples', type=int, default=500, 
                       help='Max samples per sequence')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--freeze_backbone', action='store_true', 
                       help='Freeze backbone parameters during training')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create dataset
    print("Loading dataset...")
    full_dataset = MaritimeTrackingDataset(
        args.data_dir, 
        max_samples_per_seq=args.max_samples
    )
    
    # Create model (with default 10 classes for CLS head, but we won't train it)
    print("Building model...")
    cfg = DummyConfig()
    model = build_simtrack_with_classification(cfg, num_classes=10, hidden_dim=512)
    
    # Load pretrained weights
    print(f"Loading pretrained weights from {args.pretrained_weights}...")
    model = load_pretrained_weights(model, args.pretrained_weights, strict=False)
    
    # Optionally load pre-trained classification weights
    if args.classification_weights and os.path.exists(args.classification_weights):
        print(f"Loading classification weights from {args.classification_weights}...")
        cls_checkpoint = torch.load(args.classification_weights, map_location='cpu')
        
        # Load only classification head weights
        if 'net' in cls_checkpoint:
            cls_state_dict = cls_checkpoint['net']
        else:
            cls_state_dict = cls_checkpoint
            
        # Filter classification head weights
        cls_weights = {}
        for k, v in cls_state_dict.items():
            if k.startswith(('cls_projection.', 'classifier.', 'fusion_layer.')):
                cls_weights[k] = v
        
        model.load_state_dict(cls_weights, strict=False)
        print(f"Loaded {len(cls_weights)} classification parameters")
    
    # Freeze components
    if args.freeze_backbone:
        print("Freezing backbone parameters...")
        model = freeze_backbone(model)
    
    # Freeze classification head (keep for feature extraction only)
    print("Freezing classification head...")
    model = freeze_classification_head(model)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # Move to device
    model = model.to(device)
    
    # Split dataset
    total_size = len(full_dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    
    # Create trainer
    trainer = PredictionTrainer(model, device, args.save_dir)
    
    # Training loop
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_iou = trainer.train_epoch(train_loader, optimizer, epoch)
        
        # Validate
        val_loss, val_iou = trainer.validate(val_loader, epoch)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        is_best = val_iou > trainer.best_iou
        if is_best:
            trainer.best_iou = val_iou
            
        # Save every 10 epochs or if best
        if epoch % 10 == 0 or is_best:
            trainer.save_checkpoint(epoch, optimizer, val_iou, is_best)
    
    print(f"Training completed! Best IoU: {trainer.best_iou:.3f}")


if __name__ == '__main__':
    main()