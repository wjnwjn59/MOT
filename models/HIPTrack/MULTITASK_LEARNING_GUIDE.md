# HIPTrack Multi-Task Learning Architecture

## Overview

This implementation uses **standard multi-task learning** where a single backbone produces shared features for two tasks:
1. **Bounding Box Prediction** (main task)
2. **Maritime Condition Classification** (auxiliary task)

The auxiliary classification task helps the backbone learn more robust features that benefit both tasks.

---

## Architecture

```
Input: Template + Search Images
           ↓
    ┌──────────────────┐
    │  ViT Backbone    │ ← Shared feature extractor
    │  (HIPTrack)      │
    └────────┬─────────┘
             │
       Features [B, HW, 768]
             │
     ┌───────┴────────┐
     │ Global Pooling │ (mean over spatial dims)
     └───────┬────────┘
             │
       Global Features [B, 768]
             │
    ┌────────┴────────┐
    │                 │
┌───▼────┐      ┌────▼────┐
│Box Head│      │Cls Head │
│(corner/│      │ (MLP)   │
│center) │      │         │
└───┬────┘      └────┬────┘
    │                │
    ▼                ▼
Bbox [B,4]    Classes [B,10]
```

---

## Key Components

### 1. **Shared Backbone** (`lib/models/hiptrack/hiptrack.py`)
- ViT-based feature extractor
- Processes template and search images
- Outputs features used by BOTH heads
- **No modifications needed** - uses original HIPTrack backbone

### 2. **Box Head** (existing)
- Predicts bounding boxes using Historical Prompt Network
- Uses spatial features from backbone
- **No modifications needed**

### 3. **Classification Head** (`lib/models/layers/cls_head.py`)
- **NEW**: Simple MLP for classification
- Input: Global pooled features `[B, 768]`
- Architecture:
  ```
  Linear(768 → 512) → ReLU → Dropout(0.1) → Linear(512 → 10)
  ```
- Output: Class logits `[B, 10]`

### 4. **HIPTrackCls Model** (`lib/models/hiptrack/hiptrack_cls.py`)
- Extends base HIPTrack
- Adds classification head
- **Key method**: `forward()`
  1. Calls parent `HIPTrack.forward()` to get bbox predictions
  2. Extracts backbone features (already computed)
  3. Pools features globally (mean over spatial dimensions)
  4. Passes to classification head
  5. Returns both bbox and class predictions

---

## How It Works

### Training Flow

1. **Forward Pass**:
   ```python
   # Input: template, 5 search images, cls_labels [5, B]
   
   # Step 1: Parent forward computes bbox predictions
   outputs = super().forward(...)  # Returns list of 5 outputs
   
   # Step 2: For each output, add classification
   for out in outputs:
       backbone_feat = out['backbone_feat']  # [B, HW, 768]
       
       # Extract search features (skip template patches)
       search_feat = backbone_feat[:, num_template_patches:, :]
       
       # Global pooling
       global_feat = search_feat.mean(dim=1)  # [B, 768]
       
       # Classification
       cls_logits = self.cls_head(global_feat)  # [B, 10]
       
       out['cls_logits'] = cls_logits
   ```

2. **Loss Computation** (`lib/train/actors/hiptrack_cls.py`):
   ```python
   # Compute tracking losses (GIoU, L1, Focal)
   tracking_loss = compute_tracking_losses(...)
   
   # Compute classification loss
   cls_loss = CrossEntropyLoss(cls_logits, cls_labels)
   
   # Combined loss
   total_loss = tracking_loss + cls_weight * cls_loss
   ```

3. **Backward Pass**:
   - Gradients from both tasks flow back to shared backbone
   - Backbone learns features that are good for BOTH tasks
   - This is the core of multi-task learning!

### Inference Flow

- Classification branch is **disabled** during inference
- Only bbox predictions are computed
- **No computational overhead** during tracking

---

## Configuration

### Model Config (`experiments/hiptrack/hiptrack_cls.yaml`)

```yaml
MODEL:
  HIDDEN_DIM: 768  # Shared feature dimension
  CLS_HEAD:
    NUM_CLASSES: 10  # Maritime conditions
    HIDDEN_DIM: 512  # MLP hidden dimension
    DROPOUT: 0.1     # Regularization

TRAIN:
  CLS_WEIGHT: 0.1    # Weight for classification loss
  CLS_LOSS_TYPE: "CE"  # Cross-entropy loss
  
  # Optional: Two-stage training
  TWO_STAGE: True
  STAGE1_EPOCHS: 30   # Train cls head only
  STAGE2_EPOCHS: 20   # Fine-tune everything
```

### Maritime Classification Classes

```python
0: Occlusion
1: Illumination Change
2: Scale Variation
3: Motion Blur
4: Variance in Appearance
5: Partial Visibility
6: Low Resolution
7: Background Clutter
8: Low Contrast Object
9: Normal
```

---

## Advantages of This Approach

### ✅ **Simplicity**
- Clean architecture with clear separation
- No complex fusion mechanisms
- Easy to understand and debug

### ✅ **Efficiency**
- Features computed once, used by both heads
- No additional backbone forward passes
- Minimal memory overhead (~2MB for cls head)

### ✅ **Modularity**
- Classification head is independent
- Easy to disable during inference
- Can be removed without affecting tracking

### ✅ **Multi-Task Learning Benefits**
- Auxiliary task acts as regularization
- Helps backbone learn more robust features
- Can improve tracking performance on challenging conditions

---

## Training Commands

### Standard Training
```bash
cd /home/thangdd/workspace/MOT/models/HIPTrack

python tracking/train.py \
    --script hiptrack \
    --config hiptrack_cls \
    --save_dir ./output \
    --mode single
```

### Two-Stage Training (Recommended)
```bash
# Stage 1: Train classification head only (30 epochs)
# Stage 2: Fine-tune entire model (20 epochs)
# Automatically handled by train_script_cls.py when TWO_STAGE: True

python tracking/train.py \
    --script hiptrack \
    --config hiptrack_cls \
    --save_dir ./output \
    --mode single
```

### Testing (Classification Auto-Disabled)
```bash
python tracking/test.py hiptrack_cls hiptrack_cls \
    --dataset got10k_test \
    --threads 4
```

---

## How Features Are Shared

### Feature Extraction Points

1. **Backbone Output**: `[B, HW_template + HW_search, 768]`
   - Contains features for both template and search regions
   - Stored in `out['backbone_feat']`

2. **Search Features Only**: `[B, HW_search, 768]`
   - Extract by skipping template patches
   - `search_feat = backbone_feat[:, num_template_patches:, :]`

3. **Global Pooling**: `[B, 768]`
   - Average over spatial dimensions
   - `global_feat = search_feat.mean(dim=1)`
   - Used by classification head

4. **Spatial Features**: `[B, 768, H, W]`
   - Reshaped from backbone features
   - Used by box head (through HIP module)

### Why Global Pooling?

- Classification needs **scene-level understanding**
- Pooling aggregates information from entire search region
- Creates translation-invariant representation
- Standard practice in multi-task learning

---

## Differences from Original (Fusion-Based) Implementation

| Aspect | Original (Fusion) | New (Multi-Task) |
|--------|------------------|------------------|
| **Architecture** | Separate cls branch with fusion back to box head | Shared backbone, separate heads |
| **Feature Flow** | Bidirectional (cls → box via fusion) | Unidirectional (backbone → both heads) |
| **Complexity** | High (fusion layers, dimension matching) | Low (simple pooling + MLP) |
| **Parameters** | More (fusion layers) | Fewer (no fusion) |
| **Coupling** | Tightly coupled | Loosely coupled |
| **Training** | More complex (fusion gradients) | Simpler (standard MTL) |
| **Effectiveness** | Potential for direct feature enhancement | Indirect via shared representation |

---

## Hyperparameter Tuning

### Classification Loss Weight (`CLS_WEIGHT`)
- **Default**: 0.1
- **Higher (0.5-1.0)**: More emphasis on classification
  - Use when classification is very important
  - May hurt tracking if too high
- **Lower (0.01-0.05)**: More emphasis on tracking
  - Use when tracking is primary concern
  - Classification acts as light regularization

### Hidden Dimension (`CLS_HEAD.HIDDEN_DIM`)
- **Default**: 512
- **Larger (768-1024)**: More capacity for classification
  - Use with more classes or complex patterns
  - Risk of overfitting
- **Smaller (256-384)**: Less capacity
  - Faster training
  - Better generalization

### Dropout (`CLS_HEAD.DROPOUT`)
- **Default**: 0.1
- **Higher (0.2-0.3)**: More regularization
  - Use if classification overfits
- **Lower (0.0-0.05)**: Less regularization
  - Use with small datasets

---

## Expected Performance

### Classification Accuracy
- **Goal**: 70-80% on 10-class maritime conditions
- Monitor via `Accuracy` metric in training logs
- If too low (<60%), increase `CLS_WEIGHT` or `HIDDEN_DIM`

### Tracking Performance
- **Goal**: Similar or slightly better than baseline HIPTrack
- Multi-task learning should provide regularization
- If tracking degrades, reduce `CLS_WEIGHT`

### Training Metrics
```
Epoch 1:
  Loss/total: 5.234
  Loss/giou: 0.432
  Loss/l1: 0.234
  Loss/location: 0.123
  Loss/classification: 2.301  ← Should decrease over time
  IoU: 0.678
  Accuracy: 0.423  ← Should increase over time
```

---

## Troubleshooting

### Issue: Classification accuracy stuck at ~10%
**Cause**: Random guessing (1/10 classes)
**Solution**: 
- Check if labels are loaded correctly
- Verify `cls_labels` in data pipeline
- Increase `CLS_WEIGHT` to 0.5-1.0

### Issue: Tracking performance degraded
**Cause**: Classification loss too dominant
**Solution**:
- Reduce `CLS_WEIGHT` to 0.01-0.05
- Or disable two-stage training

### Issue: Out of memory
**Cause**: Large batch size with classification head
**Solution**:
- Reduce `BATCH_SIZE` from 16 to 8
- Or reduce `CLS_HEAD.HIDDEN_DIM` from 512 to 256

---

## Files Modified

✅ Created/Modified:
1. `lib/models/layers/cls_head.py` - Simplified classification head
2. `lib/models/hiptrack/hiptrack_cls.py` - Multi-task learning model
3. `lib/train/actors/hiptrack_cls.py` - Training actor (minor fixes)
4. `lib/config/hiptrack/config_cls.py` - Removed fusion configs
5. `experiments/hiptrack/hiptrack_cls.yaml` - Updated config

✅ No changes needed:
- `lib/models/hiptrack/hiptrack.py` - Base model
- `lib/train/dataset/got10k_cls.py` - Dataset with annotations
- `lib/utils/cls_loss.py` - Loss functions
- `lib/train/train_script_cls.py` - Training script
- `lib/test/tracker/hiptrack_cls.py` - Test tracker

---

## Next Steps

1. **Test Data Pipeline**:
   ```bash
   python tracking/test_cls_annotations.py
   ```

2. **Start Training**:
   ```bash
   python tracking/train.py --script hiptrack --config hiptrack_cls --save_dir ./output --mode single
   ```

3. **Monitor Training**:
   ```bash
   tensorboard --logdir ./tensorboard/train/hiptrack/hiptrack_cls
   ```
   Watch for:
   - `Loss/classification` decreasing
   - `Accuracy` increasing
   - `IoU` stable or improving

4. **Evaluate**:
   ```bash
   python tracking/test.py hiptrack_cls hiptrack_cls --dataset got10k_test --threads 4
   ```

---

## References

- **Multi-Task Learning**: [Paper](https://arxiv.org/abs/1706.05098)
- **HIPTrack**: Historical Prompt Network for Visual Tracking
- **Maritime Tracking Challenges**: Domain-specific classification taxonomy

---

## Summary

This implementation provides **clean, efficient multi-task learning** where:
- One backbone learns features for both tracking and classification
- Simple architecture with minimal overhead
- Classification acts as auxiliary task to improve feature learning
- Easy to train, debug, and deploy

The key insight: **auxiliary tasks can improve main task performance through better shared representations**.

