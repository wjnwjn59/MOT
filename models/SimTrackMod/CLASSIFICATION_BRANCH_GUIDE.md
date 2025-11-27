# Classification Branch Integration Guide for SimTrack

## Table of Contents
1. [Overview](#overview)
2. [Architecture Design](#architecture-design)
3. [Implementation Details](#implementation-details)
4. [Training Pipeline](#training-pipeline)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)

---

## Overview

This guide documents how to add a classification branch to the SimTrack visual object tracking model. The extended architecture enables the model to simultaneously:
- **Track objects** using the original bounding box prediction head
- **Classify tracking challenges** using a new classification head

### Key Benefits
- **Multi-task learning**: Leverages shared backbone features for both tracking and classification
- **Maritime environment adaptation**: Specifically designed for maritime tracking challenges
- **Modular design**: Classification branch can be trained independently or jointly with tracking
- **Minimal overhead**: Adds only ~2.6M parameters to the base model

---

## Architecture Design

### 1. Overall Architecture

The extended SimTrack model follows a dual-stream architecture:

```
Input: [Template Image, Search Image, Template Annotation]
           ↓
    ┌──────────────┐
    │   CLIP-ViT   │  ← Backbone (can be frozen)
    │   Backbone   │
    └──────────────┘
           ↓
    ┌──────────────┐
    │  Bottleneck  │  ← Linear projection (768 → 256)
    │  (Linear)    │
    └──────────────┘
           ↓
    [CLS Token] + [Spatial Features]
           ↓
    ┌─────────────────────────────┐
    │                             │
    ▼                             ▼
┌────────────────┐    ┌───────────────────┐
│ Classification │    │   Box Prediction  │
│    Stream      │    │      Stream       │
└────────────────┘    └───────────────────┘
    │                             │
    ▼                             ▼
Projection (256→512)      Corner Predictor
ReLU + Dropout                   │
    │                             ▼
    ▼                    Bounding Box (x,y,w,h)
Classifier (512→10)
    │
    ▼
Class Logits (10 classes)
```

### 2. Component Breakdown

#### 2.1 Backbone (Shared)
- **Type**: CLIP ViT-B/16 pre-trained model
- **Input**: Template (112×112) + Search region (224×224) + Template annotation
- **Output**: Feature sequence with CLS token (first position)
- **Feature dimension**: 768 (ViT-B/16 native dimension)

#### 2.2 Bottleneck Layer
- **Purpose**: Reduce feature dimensionality for efficiency
- **Architecture**: Linear(768 → 256)
- **Position**: Between backbone and heads

#### 2.3 Classification Stream (New)

**Layer 1: Projection**
```python
self.cls_projection = nn.Linear(256, 512)  # Bottleneck output → hidden dimension
```

**Layer 2: Activation & Regularization**
```python
F.relu(cls_features)           # Non-linearity
self.dropout = nn.Dropout(0.1)  # Regularization (10% dropout)
```

**Layer 3: Classifier**
```python
self.classifier = nn.Linear(512, num_classes)  # Hidden → class logits
```

**Layer 4: Fusion (Optional)**
```python
self.fusion_layer = nn.Linear(512, 256)  # Project back to bottleneck dimension
```

#### 2.4 Box Prediction Stream (Original)
- **Architecture**: Corner-based prediction head
- **Input**: Spatial features from search region
- **Output**: Bounding box coordinates (cx, cy, w, h) normalized to [0, 1]

### 3. Feature Flow

#### 3.1 Forward Pass - Backbone Mode
```python
def forward_backbone(self, input):
    # input = [template_img, search_img, template_anno]
    output_back = self.backbone(input)  # CLIP-ViT forward
    return self.adjust(output_back)      # Apply bottleneck
```

**Feature transformations**:
1. CLIP-ViT processes concatenated template + search → `[seq_len, batch, 768]`
2. Bottleneck reduces dimension → `[seq_len, batch, 256]`
3. Reshape to transformer format → `{'feat': [HW, batch, 256]}`

#### 3.2 Forward Pass - Head Mode
```python
def forward_head(self, seq_dict, run_box_head=True, run_cls_head=True):
    feat = seq_dict[0]['feat']        # [seq_len, batch, 256]
    cls_token = feat[0]               # Extract CLS token [batch, 256]
    
    # Classification stream
    if run_cls_head:
        cls_features = self.cls_projection(cls_token)  # [batch, 512]
        cls_features = F.relu(cls_features)
        cls_features = self.dropout(cls_features)
        cls_logits = self.classifier(cls_features)     # [batch, num_classes]
        fusion_features = self.fusion_layer(cls_features)  # [batch, 256]
    
    # Box prediction stream
    if run_box_head:
        output_embed = feat  # All spatial features
        pred_boxes = self.forward_box_head(output_embed, fusion_features)
    
    return outputs
```

### 4. Classification Classes

The model supports 10 maritime tracking challenge classes:

| Class ID | Class Name | Description |
|----------|-----------|-------------|
| 0 | Motion Blur | Object blurred due to fast movement |
| 1 | Illumination Change | Lighting differences affect appearance |
| 2 | Scale Variation | Significant object size changes |
| 3 | Variance in Appearance | Pose, orientation, or deformation changes |
| 4 | Partial Visibility | Object only partially visible in frame |
| 5 | Occlusion | Object blocked by other objects |
| 6 | Background Clutter | Complex background makes object hard to distinguish |
| 7 | Low Resolution | Object appears pixelated/low quality |
| 8 | Low Contrast Object | Poor contrast with background |
| 9 | Normal | No significant tracking challenges |

---

## Implementation Details

### 1. Model Definition

**File**: `simtrack_with_classification.py`

#### Core Class Structure
```python
class SimTrackWithClassification(nn.Module):
    def __init__(self, backbone, box_head, num_classes=10, hidden_dim=512,
                 aux_loss=False, head_type="CORNER"):
        super().__init__()
        
        # Original SimTrack components
        self.backbone = backbone
        self.box_head = box_head
        self.bottleneck = nn.Linear(backbone.num_features, box_head.channel)
        
        # Classification head components
        self.cls_projection = nn.Linear(box_head.channel, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.fusion_layer = nn.Linear(hidden_dim, box_head.channel)
        self.dropout = nn.Dropout(0.1)
```

#### Key Methods

**1. Forward Backbone**
```python
def forward_backbone(self, input):
    """
    Args:
        input: [template_img, search_img, template_anno]
            - template_img: (B, 3, 112, 112)
            - search_img: (B, 3, 224, 224)
            - template_anno: (B, 4) - [cx, cy, w, h] normalized
    
    Returns:
        dict: {'feat': (HW, B, C)} - feature sequence
    """
    output_back = self.backbone(input)
    return self.adjust(output_back)
```

**2. Forward Head**
```python
def forward_head(self, seq_dict, run_box_head=True, run_cls_head=True):
    """
    Args:
        seq_dict: List containing backbone features
        run_box_head: Whether to run box prediction
        run_cls_head: Whether to run classification
    
    Returns:
        dict: {
            'cls_logits': (B, num_classes),     # if run_cls_head
            'pred_boxes': (B, N, 4),            # if run_box_head
            'cls_features': (B, hidden_dim)     # if run_cls_head
        }
    """
```

**3. Adjust Features**
```python
def adjust(self, output_back):
    """Apply bottleneck and reshape for transformer"""
    src_feat = output_back
    feat = self.bottleneck(src_feat)  # Reduce dimension
    feat_vec = feat.flatten(2).permute(1, 0, 2)  # HWxBxC
    return {"feat": feat_vec}
```

### 2. Model Building

```python
def build_simtrack_with_classification(cfg, num_classes=10, hidden_dim=512):
    """
    Build SimTrack model with classification head
    
    Args:
        cfg: Configuration object with model settings
        num_classes: Number of classification classes (default: 10)
        hidden_dim: Hidden dimension for classification layers (default: 512)
    
    Returns:
        SimTrackWithClassification model
    """
    backbone = build_backbone_simtrack(cfg)
    box_head = build_box_head(cfg)
    
    model = SimTrackWithClassification(
        backbone=backbone,
        box_head=box_head,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        aux_loss=cfg.TRAIN.DEEP_SUPERVISION,
        head_type=cfg.MODEL.HEAD_TYPE
    )
    
    return model
```

### 3. Weight Management

#### Loading Pretrained Weights
```python
def load_pretrained_weights(model, weight_path, strict=False):
    """
    Load pretrained SimTrack weights into extended model
    
    Strategy:
    - Loads backbone and box_head weights (tracking components)
    - Classification head weights are randomly initialized
    - Safely handles shape mismatches and missing keys
    """
    checkpoint = torch.load(weight_path, map_location='cpu')
    model_dict = model.state_dict()
    
    # Filter to only matching keys and shapes
    filtered_dict = {}
    for k, v in checkpoint.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered_dict[k] = v
    
    model.load_state_dict(filtered_dict, strict=strict)
    return model
```

#### Freezing Backbone
```python
def freeze_backbone(model):
    """
    Freeze backbone and bottleneck for classification-only training
    
    This is useful for:
    - Fast fine-tuning on new classification tasks
    - Preventing catastrophic forgetting of tracking abilities
    - Reducing GPU memory requirements
    """
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    for param in model.bottleneck.parameters():
        param.requires_grad = False
    
    # Only classification head remains trainable
```

---

## Training Pipeline

### 1. Dataset Preparation

**File**: `finetune_classification.py`

#### Dataset Class
```python
class MaritimeClassificationDataset(Dataset):
    """
    Dataset for maritime environment classification
    
    Data Structure:
        data_dir/
        ├── sequence_1/
        │   ├── template_crop.jpg          # Target object template
        │   ├── annotations.jsonl          # Per-frame annotations
        │   └── frames/
        │       ├── frame_001.jpg
        │       ├── frame_002.jpg
        │       └── ...
        ├── sequence_2/
        └── ...
    
    JSONL Format:
    {
        "sequence_name": "1-Boat",
        "frame_file": "img0001.jpg",
        "template_bbox": [x1, y1, x2, y2],
        "ground_truth_bbox": [x1, y1, x2, y2],
        "vlm_response": {
            "motion_blur": {"flag": 0, "confidence": 0.2},
            "illu_change": {"flag": 1, "confidence": 0.8},
            ...
        },
        "cv_response": {
            "scale_variation": 0,
            "low_res": 0,
            ...
        }
    }
    """
```

#### Key Methods

**1. Build Class Mapping**
```python
def _build_class_mapping(self):
    """Map class names to indices"""
    vlm_classes = ["motion_blur", "illu_change", "variance_appear",
                   "partial_visibility", "background_clutter", "occlusion"]
    cv_classes = ["scale_variation", "low_res", "low_contrast"]
    all_classes = vlm_classes + cv_classes + ["normal"]
    return {cls: idx for idx, cls in enumerate(all_classes)}
```

**2. Determine Class Label**
```python
def _get_difficulty_class(self, vlm_response, cv_response):
    """
    Extract primary difficulty class from responses
    
    Priority:
    1. VLM response (vision-language model annotation)
    2. CV response (computer vision metrics)
    3. Default to "normal"
    """
    # Check VLM flags first
    for key, value in vlm_response.items():
        if value.get('flag', 0) == 1:
            return key
    
    # Check CV flags
    for key, value in cv_response.items():
        if value == 1:
            return key
    
    return "normal"
```

**3. Data Loading**
```python
def __getitem__(self, idx):
    """
    Returns:
        dict: {
            'template_img': (3, 112, 112) - Cropped target template
            'search_img': (3, 224, 224) - Search region frame
            'template_anno': (4,) - Normalized bbox [cx, cy, w, h]
            'gt_bbox': (4,) - Ground truth bbox
            'class_label': int - Class index (0-9)
            'class_name': str - Class name
            'sequence_name': str - Sequence identifier
        }
    """
```

### 2. Training Configuration

#### Hyperparameters
```python
# Model
num_classes = 10          # Number of difficulty classes
hidden_dim = 512          # Classification hidden layer size

# Training
batch_size = 8            # Adjust based on GPU memory
epochs = 50               # Fine-tuning epochs
learning_rate = 1e-4      # Adam learning rate
weight_decay = 1e-4       # L2 regularization

# Data
val_split = 0.2           # Validation set ratio
max_samples_per_seq = 500 # Limit frames per sequence
```

#### Class Imbalance Handling
```python
def compute_class_weights(dataset, num_classes):
    """
    Compute inverse frequency weights for imbalanced classes
    
    Formula:
        weight_i = total_samples / (num_classes * count_i)
    
    This gives higher weight to rare classes during training.
    """
    class_counts = np.zeros(num_classes)
    for sample in dataset.samples:
        class_counts[sample['class_label']] += 1
    
    weights = total_samples / (num_classes * class_counts)
    return weights / weights.sum() * num_classes
```

### 3. Training Loop

**File**: `finetune_classification.py`

```python
class ClassificationTrainer:
    def train_epoch(self, dataloader, optimizer, epoch):
        """
        Training procedure for one epoch
        
        For each batch:
        1. Forward backbone (extract features)
        2. Forward classification head
        3. Compute cross-entropy loss (weighted)
        4. Backward pass and optimize
        5. Track accuracy and losses
        """
        self.model.train()
        
        for batch in dataloader:
            # Move to device
            template_img = batch['template_img'].to(device)
            search_img = batch['search_img'].to(device)
            template_anno = batch['template_anno'].to(device)
            class_labels = batch['class_label'].to(device)
            
            # Forward pass
            input_data = [template_img, search_img, template_anno]
            backbone_output = self.model.forward_backbone(input_data)
            
            seq_dict = [backbone_output]
            head_output = self.model.forward_head(
                seq_dict, 
                run_box_head=False,   # Only classification
                run_cls_head=True
            )
            
            # Compute loss
            cls_logits = head_output['cls_logits']
            loss = self.classification_loss(cls_logits, class_labels)
            
            # Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### 4. Loss Functions

#### Classification Loss
```python
# Weighted Cross-Entropy for imbalanced classes
class_weights = compute_class_weights(dataset, num_classes)
class_weights = torch.tensor(class_weights).to(device)

classification_loss = nn.CrossEntropyLoss(weight=class_weights)
```

#### Combined Loss (if training both tasks)
```python
# For joint training of tracking + classification
total_loss = (
    lambda_giou * giou_loss +      # Tracking: GIoU loss
    lambda_l1 * l1_loss +          # Tracking: L1 loss
    lambda_cls * cls_loss          # Classification: CE loss
)

# Typical weights:
# lambda_giou = 2.0
# lambda_l1 = 5.0
# lambda_cls = 1.0
```

### 5. Validation and Metrics

```python
def validate(self, dataloader, epoch):
    """
    Validation with detailed metrics
    
    Metrics:
    - Overall accuracy
    - Per-class accuracy
    - Confusion matrix
    - Loss
    """
    self.model.eval()
    
    class_correct = {class_name: 0 for class_name in classes}
    class_total = {class_name: 0 for class_name in classes}
    
    with torch.no_grad():
        for batch in dataloader:
            # Forward pass
            outputs = self.model(...)
            predictions = torch.argmax(outputs['cls_logits'], dim=1)
            
            # Accumulate statistics
            for i, (pred, target) in enumerate(zip(predictions, labels)):
                class_name = class_names[target]
                class_total[class_name] += 1
                if pred == target:
                    class_correct[class_name] += 1
    
    # Print per-class accuracy
    for class_name in class_correct:
        accuracy = class_correct[class_name] / class_total[class_name]
        print(f"{class_name}: {accuracy:.2%}")
```

### 6. Training Script Usage

```bash
# Basic training (frozen backbone)
python finetune_classification.py \
    --data_dir /path/to/maritime_annotations \
    --pretrained_weights sim-vit-b-16.pth \
    --freeze_backbone \
    --batch_size 8 \
    --epochs 50 \
    --lr 1e-4

# Full fine-tuning (train everything)
python finetune_classification.py \
    --data_dir /path/to/maritime_annotations \
    --pretrained_weights sim-vit-b-16.pth \
    --batch_size 4 \
    --epochs 100 \
    --lr 1e-5

# Resume from checkpoint
python finetune_classification.py \
    --data_dir /path/to/maritime_annotations \
    --resume checkpoints/classification/simtrack_classification_ep0030.pth.tar
```

---

## Usage Examples

### 1. Basic Inference

```python
import torch
from simtrack_with_classification import (
    build_simtrack_with_classification,
    load_pretrained_weights,
    DummyConfig
)

# Build model
cfg = DummyConfig()
model = build_simtrack_with_classification(cfg, num_classes=10, hidden_dim=512)

# Load weights
model = load_pretrained_weights(model, "sim-vit-b-16.pth", strict=False)
model = model.cuda()
model.eval()

# Prepare inputs
template_img = torch.randn(1, 3, 112, 112).cuda()
search_img = torch.randn(1, 3, 224, 224).cuda()
template_anno = torch.tensor([[0.5, 0.5, 0.2, 0.2]]).cuda()  # [cx, cy, w, h]

# Forward pass
with torch.no_grad():
    # Extract features
    input_data = [template_img, search_img, template_anno]
    backbone_output = model.forward_backbone(input_data)
    
    # Run both heads
    seq_dict = [backbone_output]
    outputs = model.forward_head(seq_dict, run_box_head=True, run_cls_head=True)
    
    # Get predictions
    pred_boxes = outputs['pred_boxes']      # (1, 1, 4) - [cx, cy, w, h]
    cls_logits = outputs['cls_logits']      # (1, 10)
    predicted_class = torch.argmax(cls_logits, dim=1)

print(f"Predicted bbox: {pred_boxes}")
print(f"Predicted class: {predicted_class.item()}")
```

### 2. Classification Only

```python
# For pure classification (no tracking)
model = build_simtrack_with_classification(cfg)
model = load_pretrained_weights(model, "classification_weights.pth")
model.eval()

# Forward pass
backbone_output = model.forward_backbone([template, search, anno])
outputs = model.forward_head(
    [backbone_output],
    run_box_head=False,   # Skip tracking
    run_cls_head=True     # Only classification
)

probabilities = F.softmax(outputs['cls_logits'], dim=1)
print(f"Class probabilities: {probabilities}")
```

### 3. Tracking with Challenge Detection

```python
# Production scenario: Track object and detect challenges
model.eval()

for frame in video_frames:
    # Prepare inputs
    search_img = preprocess(frame)
    
    # Inference
    with torch.no_grad():
        backbone_out = model.forward_backbone([template, search_img, template_anno])
        outputs = model.forward_head([backbone_out], 
                                     run_box_head=True, 
                                     run_cls_head=True)
    
    # Extract predictions
    bbox = outputs['pred_boxes'][0, 0]  # Current frame bbox
    cls_id = torch.argmax(outputs['cls_logits'], dim=1).item()
    
    # Handle based on difficulty
    if cls_id == 5:  # Occlusion
        print("Warning: Occlusion detected, confidence may be low")
    elif cls_id == 3:  # Motion blur
        print("Warning: Motion blur, consider temporal smoothing")
    
    # Update tracking
    current_bbox = bbox
```

### 4. Fine-tuning on Custom Dataset

```python
from finetune_classification import MaritimeClassificationDataset, ClassificationTrainer

# Load your dataset
dataset = MaritimeClassificationDataset(
    data_dir="path/to/your/annotations",
    max_samples_per_seq=1000
)

# Split into train/val
train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

# Create dataloaders
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

# Build model
model = build_simtrack_with_classification(cfg, num_classes=len(dataset.class_mapping))
model = load_pretrained_weights(model, "pretrained.pth")

# Optionally freeze backbone
model = freeze_backbone(model)

# Setup training
optimizer = AdamW(model.parameters(), lr=1e-4)
trainer = ClassificationTrainer(model, device, save_dir="checkpoints/")

# Train
for epoch in range(50):
    train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, epoch)
    val_loss, val_acc = trainer.validate(val_loader, epoch)
    
    if val_acc > trainer.best_accuracy:
        trainer.save_checkpoint(epoch, optimizer, val_acc, is_best=True)
```

---

## Best Practices

### 1. Model Architecture

#### Choose Appropriate Hidden Dimension
```python
# For 10 classes, hidden_dim=512 works well
# For more classes (e.g., 50+), consider hidden_dim=1024
model = build_simtrack_with_classification(
    cfg,
    num_classes=10,
    hidden_dim=512  # Balance between capacity and efficiency
)
```

#### Regularization
```python
# Use dropout to prevent overfitting
self.dropout = nn.Dropout(0.1)  # 10% dropout rate

# Apply weight decay
optimizer = AdamW(model.parameters(), 
                  lr=1e-4, 
                  weight_decay=1e-4)  # L2 regularization
```

### 2. Training Strategy

#### Two-Stage Training (Recommended)
```python
# Stage 1: Freeze backbone, train classification head only (fast)
model = freeze_backbone(model)
train(model, epochs=30, lr=1e-3)

# Stage 2: Fine-tune entire model (slow but better performance)
model = unfreeze_backbone(model)
train(model, epochs=20, lr=1e-5)  # Lower learning rate
```

#### Learning Rate Schedule
```python
# Use step decay
scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

# Or cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
```

### 3. Data Handling

#### Handle Class Imbalance
```python
# Option 1: Weighted loss
class_weights = compute_class_weights(dataset, num_classes)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Option 2: Over-sampling rare classes
sampler = WeightedRandomSampler(weights, num_samples=len(dataset))
dataloader = DataLoader(dataset, sampler=sampler, batch_size=8)
```

#### Data Augmentation
```python
# Add augmentation for robustness
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

### 4. Evaluation

#### Comprehensive Metrics
```python
from sklearn.metrics import confusion_matrix, classification_report

def evaluate(model, dataloader):
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        outputs = model(batch)
        preds = torch.argmax(outputs['cls_logits'], dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['class_label'].cpu().numpy())
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Per-class metrics
    report = classification_report(all_labels, all_preds, 
                                   target_names=class_names)
    
    return cm, report
```

### 5. Deployment

#### Model Export
```python
# Export to TorchScript for production
model.eval()
example_inputs = [
    torch.randn(1, 3, 112, 112),
    torch.randn(1, 3, 224, 224),
    torch.tensor([[0.5, 0.5, 0.2, 0.2]])
]

traced_model = torch.jit.trace(model.forward_backbone, [example_inputs])
traced_model.save("simtrack_classification.pt")
```

#### Optimization for Inference
```python
# Mixed precision for faster inference
with torch.cuda.amp.autocast():
    outputs = model(inputs)

# Batch processing for throughput
def batch_inference(model, frames, batch_size=32):
    results = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        with torch.no_grad():
            outputs = model(batch)
        results.append(outputs)
    return torch.cat(results)
```

### 6. Monitoring and Debugging

#### Gradient Monitoring
```python
# Check for vanishing/exploding gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm()
        if grad_norm > 10:
            print(f"Large gradient in {name}: {grad_norm}")
```

#### Feature Visualization
```python
# Visualize classification features
def visualize_features(model, sample):
    backbone_out = model.forward_backbone(sample)
    head_out = model.forward_head([backbone_out], run_cls_head=True)
    
    cls_features = head_out['cls_features']  # (batch, 512)
    
    # t-SNE visualization
    from sklearn.manifold import TSNE
    features_2d = TSNE(n_components=2).fit_transform(cls_features.cpu())
    
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels)
    plt.colorbar()
    plt.show()
```

---

## Troubleshooting

### Common Issues

**1. NaN Loss**
```python
# Check for invalid inputs
assert not torch.isnan(images).any()
assert not torch.isinf(images).any()

# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**2. Poor Classification Accuracy**
- Verify class labels are correct
- Check class distribution (handle imbalance)
- Increase model capacity (hidden_dim)
- Add more training data
- Use data augmentation

**3. GPU Out of Memory**
```python
# Reduce batch size
batch_size = 4  # Instead of 8

# Use gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## References

### Key Files
- `simtrack_with_classification.py` - Extended model definition
- `finetune_classification.py` - Training script
- `mvtd_classification.py` - Dataset annotation pipeline
- `lib/models/stark/simtrack.py` - Original SimTrack model
- `lib/models/stark/backbone.py` - CLIP-ViT backbone
- `lib/models/stark/head.py` - Box prediction heads

### Related Papers
- **SimTrack**: "Backbone is All Your Need" (ECCV 2022)
- **CLIP**: "Learning Transferable Visual Models" (ICML 2021)
- **Vision Transformer**: "An Image is Worth 16x16 Words" (ICLR 2021)

---

## Summary

This guide covered:
✅ Architecture design with dual-stream heads  
✅ Implementation details for classification branch  
✅ Complete training pipeline with class imbalance handling  
✅ Usage examples for various scenarios  
✅ Best practices for training and deployment  

The classification branch extends SimTrack's capabilities while maintaining its efficient tracking performance, making it ideal for maritime and challenging tracking scenarios.
