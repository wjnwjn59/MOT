"""
Fine-tuning script for SimTrack with Classification Head
Trains on maritime environment classification annotations
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
from lib.utils.box_ops import box_xywh_to_xyxy
from lib.train.admin import multigpu


class MaritimeClassificationDataset(Dataset):
    """Dataset for maritime environment classification"""
    
    def __init__(self, data_dir, transform=None, max_samples_per_seq=None):
        self.data_dir = data_dir
        self.transform = transform
        self.max_samples_per_seq = max_samples_per_seq
        
        # Load all annotation files
        self.samples = []
        self.class_mapping = self._build_class_mapping()
        self._load_annotations()
        print("Data path: ", self.data_dir)
        print(f"Loaded {len(self.samples)} samples from {len(os.listdir(data_dir))} sequences")
        print(f"Class mapping: {self.class_mapping}")
        
    def _build_class_mapping(self):
        """Build mapping from class names to indices"""
        # Based on the jsonl structure, we have these classes
        vlm_classes = [
            "motion_blur", "illu_change", "variance_appear", 
            "partial_visibility", "background_clutter", "occlusion"
        ]
        cv_classes = ["scale_variation", "low_res", "low_contrast"]
        
        # Combined classes + normal class
        all_classes = vlm_classes + cv_classes + ["normal"]
        
        return {cls: idx for idx, cls in enumerate(all_classes)}
    
    def _get_difficulty_class(self, vlm_response, cv_response):
        """Determine the difficulty class based on vlm and cv responses"""
        # Priority 1: Check VLM response
        for key, value in vlm_response.items():
            if isinstance(value, dict) and value.get('flag', 0) == 1:
                return key
        
        # Priority 2: Check CV response  
        for key, value in cv_response.items():
            if value == 1:
                return key
                
        # Default: normal class
        return "normal"
    
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
                        
                        # Get difficulty class
                        difficulty_class = self._get_difficulty_class(
                            data['vlm_response'], data['cv_response']
                        )
                        
                        if difficulty_class not in self.class_mapping:
                            print(f"Unknown class: {difficulty_class}")
                            continue
                        
                        # Construct sample
                        sample = {
                            'sequence_name': data['sequence_name'],
                            'frame_file': data['frame_file'], 
                            'template_crop': template_crop,
                            'template_bbox': data['template_bbox'],
                            'ground_truth_bbox': data['ground_truth_bbox'],
                            'class_label': self.class_mapping[difficulty_class],
                            'class_name': difficulty_class,
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
            "../..",
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
            'class_label': torch.tensor(sample['class_label'], dtype=torch.long),
            'class_name': sample['class_name'],
            'sequence_name': sample['sequence_name']
        }
    
    def _preprocess_image(self, img, size):
        """Preprocess image to tensor"""
        # Resize
        img = img.resize((size, size), Image.BILINEAR)
        
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


def compute_class_weights(dataset, num_classes):
    """Compute class weights for imbalanced dataset"""
    class_counts = np.zeros(num_classes)
    
    for sample in dataset.samples:
        class_label = sample['class_label']
        class_counts[class_label] += 1
    
    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)
    
    # Compute inverse frequency weights
    total_samples = len(dataset.samples)
    weights = total_samples / (num_classes * class_counts)
    
    # Normalize weights
    weights = weights / np.sum(weights) * num_classes
    
    return weights


class ClassificationTrainer:
    """Trainer for classification fine-tuning"""
    
    def __init__(self, model, device, save_dir, class_weights=None):
        self.model = model
        self.device = device 
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss functions with class weights for imbalanced data
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
            print(f"Using class weights: {class_weights}")
        self.classification_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        # Metrics
        self.best_accuracy = 0.0
        self.train_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, dataloader, optimizer, epoch):
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            template_img = batch['template_img'].to(self.device)
            search_img = batch['search_img'].to(self.device)  
            template_anno = batch['template_anno'].to(self.device)
            class_labels = batch['class_label'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            try:
                # Backbone forward
                input_data = [template_img, search_img, template_anno]
                backbone_output = self.model.forward_backbone(input_data)
                
                # Head forward (only classification)
                seq_dict = [backbone_output]
                head_output = self.model.forward_head(
                    seq_dict, run_box_head=False, run_cls_head=True
                )
                
                # Get classification logits
                cls_logits = head_output['cls_logits']
                
                # Ensure float32 type consistency
                cls_logits = cls_logits.float()
                
                # Compute loss
                cls_loss = self.classification_loss(cls_logits, class_labels)
                loss = cls_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                total_cls_loss += cls_loss.item()
                
                # Accuracy
                predicted = torch.argmax(cls_logits, dim=1)
                correct_predictions += (predicted == class_labels).sum().item()
                total_samples += class_labels.size(0)
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
                
            # Update progress bar
            if batch_idx % 10 == 0:
                current_acc = correct_predictions / max(total_samples, 1) * 100
                pbar.set_postfix({
                    'Loss': f'{total_loss/(batch_idx+1):.4f}',
                    'Cls_Loss': f'{total_cls_loss/(batch_idx+1):.4f}', 
                    'Acc': f'{current_acc:.2f}%'
                })
        
        # Epoch statistics
        avg_loss = total_loss / len(dataloader)
        avg_cls_loss = total_cls_loss / len(dataloader)
        accuracy = correct_predictions / max(total_samples, 1) * 100
        
        self.train_losses.append(avg_loss)
        
        print(f'Epoch {epoch} - Loss: {avg_loss:.4f}, Cls_Loss: {avg_cls_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        return avg_loss, accuracy
    
    def validate(self, dataloader, epoch):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        class_correct = {}
        class_total = {}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                # Move to device
                template_img = batch['template_img'].to(self.device)
                search_img = batch['search_img'].to(self.device)
                template_anno = batch['template_anno'].to(self.device) 
                class_labels = batch['class_label'].to(self.device)
                class_names = batch['class_name']
                
                try:
                    # Forward pass
                    input_data = [template_img, search_img, template_anno]
                    backbone_output = self.model.forward_backbone(input_data)
                    
                    seq_dict = [backbone_output]
                    head_output = self.model.forward_head(
                        seq_dict, run_box_head=False, run_cls_head=True
                    )
                    
                    cls_logits = head_output['cls_logits']
                    
                    # Ensure float32 type consistency
                    cls_logits = cls_logits.float()
                    
                    # Compute loss
                    cls_loss = self.classification_loss(cls_logits, class_labels)
                    total_loss += cls_loss.item()
                    
                    # Accuracy
                    predicted = torch.argmax(cls_logits, dim=1)
                    correct_predictions += (predicted == class_labels).sum().item()
                    total_samples += class_labels.size(0)
                    
                    # Per-class accuracy
                    for i, class_name in enumerate(class_names):
                        if class_name not in class_correct:
                            class_correct[class_name] = 0
                            class_total[class_name] = 0
                        
                        class_total[class_name] += 1
                        if predicted[i] == class_labels[i]:
                            class_correct[class_name] += 1
                            
                except Exception as e:
                    print(f"Error in validation: {e}")
                    continue
        
        # Statistics
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / max(total_samples, 1) * 100
        
        self.val_accuracies.append(accuracy)
        
        # Per-class accuracy
        print(f'Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        print("Per-class accuracy:")
        for class_name in class_correct:
            if class_total[class_name] > 0:
                acc = class_correct[class_name] / class_total[class_name] * 100
                print(f'  {class_name}: {acc:.2f}% ({class_correct[class_name]}/{class_total[class_name]})')
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, optimizer, accuracy, is_best=False):
        """Save model checkpoint"""
        net = self.model.module if multigpu.is_multi_gpu(self.model) else self.model
        
        state = {
            'epoch': epoch,
            'net_type': type(net).__name__,
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'accuracy': accuracy,
            'best_accuracy': self.best_accuracy,
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'simtrack_classification_ep{epoch:04d}.pth.tar')
        torch.save(state, checkpoint_path)
        print(f'Saved checkpoint: {checkpoint_path}')
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'simtrack_classification_best.pth.tar')
            torch.save(state, best_path)
            print(f'Saved best model: {best_path}')


def main():
    parser = argparse.ArgumentParser(description='Fine-tune SimTrack Classification Head')
    parser.add_argument('--data_dir', type=str, 
                       default='../data/train_maritime_env_clf_annts',
                       help='Path to maritime classification data')
    parser.add_argument('--pretrained_weights', type=str,
                       default='./checkpoints/sim-vit-b-16.pth',
                       help='Path to pretrained weights')
    parser.add_argument('--save_dir', type=str,
                       default='./checkpoints/classification',
                       help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--max_samples', type=int, default=500, 
                       help='Max samples per sequence')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--freeze_backbone', action='store_true', 
                       help='Freeze backbone parameters during training')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create dataset first to determine number of classes
    print("Loading dataset...")
    full_dataset = MaritimeClassificationDataset(
        args.data_dir, 
        max_samples_per_seq=args.max_samples
    )
    
    num_classes = len(full_dataset.class_mapping)
    print(f"Number of classes: {num_classes}")
    
    # Compute class weights for imbalanced dataset
    class_weights = compute_class_weights(full_dataset, num_classes)
    print(f"Class weights: {class_weights}")
    
    # Create model
    print("Building model...")
    cfg = DummyConfig()
    model = build_simtrack_with_classification(cfg, num_classes=num_classes, hidden_dim=512)
    
    # Load pretrained weights
    print(f"Loading pretrained weights from {args.pretrained_weights}...")
    model = load_pretrained_weights(model, args.pretrained_weights, strict=False)
    
    # Optionally freeze backbone to only train classification head
    if args.freeze_backbone:
        print("Freezing backbone parameters...")
        model = freeze_backbone(model)
    else:
        print("Training all parameters (backbone + classification head)")
    
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
    
    # Create trainer with class weights
    trainer = ClassificationTrainer(model, device, args.save_dir, class_weights=class_weights)
    
    # Training loop
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, epoch)
        
        # Validate
        val_loss, val_acc = trainer.validate(val_loader, epoch)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        is_best = val_acc > trainer.best_accuracy
        if is_best:
            trainer.best_accuracy = val_acc
            
        # Save every 10 epochs or if best
        if epoch % 10 == 0 or is_best:
            trainer.save_checkpoint(epoch, optimizer, val_acc, is_best)
    
    print(f"Training completed! Best accuracy: {trainer.best_accuracy:.2f}%")


if __name__ == '__main__':
    main()