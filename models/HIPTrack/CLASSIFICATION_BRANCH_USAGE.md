# HIPTrack with Classification Branch - Usage Guide

```bash
# Test annotations
python tracking/test_cls_annotations.py

# Training
python tracking/train.py --script hiptrack_cls --config hiptrack_cls --save_dir ./output --mode single

python lib/train/run_training.py --script hiptrack --config hiptrack_cls --save_dir ./output --use_lmdb 0 --script_prv None --config_prv baseline --distill 0 --script_teacher None --config_teacher None --use_wandb 0

# Testing (cls auto-disabled)
python tracking/test.py hiptrack_cls hiptrack_cls --dataset got10k_test --threads 4

# Convert checkpoint
python tracking/convert_cls_to_tracking.py --input <input.pth.tar> --output <output.pth.tar>
```


## Overview

This implementation adds an auxiliary classification branch to HIPTrack for maritime tracking challenges. The classification branch:
- Is used during training to improve feature learning
- Classifies 10 types of tracking challenges (Occlusion, Illumination Change, Scale Variation, Motion Blur, etc.)
- Is automatically disabled during inference
- Does not affect tracking performance

## Files Created

### Core Components
1. **lib/models/layers/cls_head.py** - Classification head implementation
2. **lib/models/hiptrack/hiptrack_cls.py** - HIPTrack model with classification
3. **lib/utils/cls_loss.py** - Classification loss functions
4. **lib/config/hiptrack/config_cls.py** - Configuration with classification settings
5. **lib/train/dataset/got10k_cls.py** - Dataset with classification labels
6. **lib/train/actors/hiptrack_cls.py** - Training actor with classification
7. **lib/train/train_script_cls.py** - Training script
8. **lib/test/tracker/hiptrack_cls.py** - Test tracker (classification disabled)
9. **lib/test/parameter/hiptrack_cls.py** - Test parameters
10. **experiments/hiptrack/hiptrack_cls.yaml** - YAML configuration file
11. **tracking/convert_cls_to_tracking.py** - Checkpoint conversion tool

### Modified Files
- **lib/train/data/sampler.py** - Added classification label handling

## Classification Classes

The model classifies frames into 10 categories based on tracking challenges:

1. **Occlusion** (0) - Object is occluded by other objects
2. **Illumination Change** (1) - Lighting conditions change
3. **Scale Variation** (2) - Object size changes significantly
4. **Motion Blur** (3) - Fast motion causes blur
5. **Variance in Appearance** (4) - Object appearance changes
6. **Partial Visibility** (5) - Object is partially visible
7. **Low Resolution** (6) - Object has low resolution
8. **Background Clutter** (7) - Complex background
9. **Low Contrast Object** (8) - Object has low contrast with background
10. **Normal** (9) - No special challenges

### Classification Logic

The classification label is determined from JSONL annotations with the following priority:

1. **VLM Response (Priority 1)**: Check `vlm_response` fields first
   - If ANY VLM flag is 1, use that class immediately
   - Order checked: occlusion → motion_blur → illu_change → partial_visibility → variance_appear → background_clutter

2. **CV Response (Priority 2)**: If no VLM response found, check `cv_response` fields
   - If multiple CV flags are 1, **randomly select one**
   - Options: scale_variation, low_res, low_contrast

3. **Default**: If no flags are set, classify as "Normal" (class 9)

## Training

### Single GPU Training

```bash
cd /home/thinhnp/MOT/models/HIPTrack

python tracking/train.py \
    --script hiptrack \
    --config hiptrack_cls \
    --save_dir ./output \
    --mode single
```

### Multi-GPU Training

```bash
python tracking/train.py \
    --script hiptrack \
    --config hiptrack_cls \
    --save_dir ./output \
    --mode multiple \
    --nproc_per_node 4
```

### Configuration

Key parameters in `experiments/hiptrack/hiptrack_cls.yaml`:

```yaml
MODEL:
  CLS_HEAD:
    NUM_CLASSES: 10          # Number of classification classes
    HIDDEN_DIM: 512          # Hidden dimension
    POOLING: 'avg'           # Pooling method: avg, max, attention
    DROPOUT: 0.3             # Dropout rate

TRAIN:
  CLS_WEIGHT: 1.0            # Classification loss weight
  CLS_LOSS_TYPE: "CE"        # Loss type: CE, FOCAL, LABEL_SMOOTH
  FREEZE_CLS_EPOCH: -1       # Freeze cls after epoch (-1 = never)
  
DATA:
  TRAIN:
    CLS_ANN_DIR: "/home/thinhnp/MOT/data/train_maritime_env_clf_annts"
```

## Testing/Inference

During testing, the classification branch is automatically disabled.

### Test on GOT10K

```bash
python tracking/test.py hiptrack_cls hiptrack_cls \
    --dataset got10k_test \
    --threads 4
```

### Test on LaSOT

```bash
python tracking/test.py hiptrack_cls hiptrack_cls \
    --dataset lasot \
    --threads 4
```

## Convert Checkpoint (Optional)

To create a checkpoint without the classification branch:

```bash
python tracking/convert_cls_to_tracking.py \
    --input ./output/checkpoints/train/hiptrack/hiptrack_cls/HIPTrack_ep0300.pth.tar \
    --output ./pretrained_models/HIPTrack_tracking_only.pth.tar
```

## Classification Annotation Format

Classification annotations are stored in JSONL format:

**Location**: `/home/thinhnp/MOT/data/train_maritime_env_clf_annts/{sequence_name}/{sequence_name}.jsonl`

**Example**: `/home/thinhnp/MOT/data/train_maritime_env_clf_annts/1-Ship/1-Ship.jsonl`

**Format** (one JSON object per line):
```json
{
  "sequence_name": "1-Ship",
  "frame_id": 1,
  "frame_file": "00000001.jpg",
  "cv_response": {
    "scale_variation": 0,
    "low_res": 0,
    "low_contrast": 0
  },
  "vlm_response": {
    "motion_blur": {"flag": 0, "conf": 0.0},
    "illu_change": {"flag": 0, "conf": 0.0},
    "variance_appear": {"flag": 0, "conf": 0.0},
    "partial_visibility": {"flag": 1, "conf": 1.0},
    "background_clutter": {"flag": 0, "conf": 0.0},
    "occlusion": {"flag": 0, "conf": 0.0}
  },
  "ground_truth_bbox": [554.0, 184.0, 683.0, 358.0]
}
```

**Label Selection Logic:**
- Above example: `partial_visibility` flag is 1 in VLM → Class 5 (Partial Visibility)
- If VLM has multiple flags: Use first one found in priority order
- If only CV has flags: If `scale_variation=1` and `low_res=1` → Randomly pick one (Class 2 or 6)
- If no flags: Class 9 (Normal)

## Hyperparameter Tuning

### CLS_WEIGHT
- Start with 0.5-1.0
- Increase if classification accuracy is low
- Decrease if tracking performance degrades

### CLS_LOSS_TYPE
- **CE**: For balanced classes
- **FOCAL**: For imbalanced classes
- **LABEL_SMOOTH**: To reduce overfitting

### FREEZE_CLS_EPOCH
- Set to -1 to never freeze (train throughout)
- Set to 200-250 to freeze in later epochs
- Helps stabilize tracking performance

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir ./tensorboard/train/hiptrack/hiptrack_cls
```

### Training Metrics

The training will log:
- `Loss/total` - Total loss (tracking + classification)
- `Loss/giou` - GIoU loss
- `Loss/l1` - L1 loss
- `Loss/location` - Focal loss
- `Loss/classification` - Classification loss
- `IoU` - Tracking IoU
- `Accuracy` - Classification accuracy

## Troubleshooting

### Issue: NaN Loss
**Solution**: Reduce learning rate or CLS_WEIGHT

### Issue: Low Classification Accuracy
**Solutions**:
1. Check annotation files are correctly formatted
2. Increase CLS_WEIGHT
3. Try different loss type (FOCAL for imbalanced data)
4. Increase training epochs

### Issue: Tracking Performance Degraded
**Solutions**:
1. Decrease CLS_WEIGHT
2. Set FREEZE_CLS_EPOCH to freeze earlier
3. Check if classification labels are correct

### Issue: Missing Classification Annotations
**Behavior**: Dataset will use ignore index (-100) for frames without annotations
**Impact**: These frames won't contribute to classification loss

## Architecture Details

### Classification Head
- Input: Search region features `[B, HW, C]`
- Pooling: Adaptive pooling (avg/max/attention)
- MLP: Two-layer with ReLU and Dropout
- Output: Class logits `[B, 10]`

### Integration
- Classification head receives features from ViT backbone
- Only search region features are used (template features ignored)
- Gradients from classification loss help improve backbone features
- During inference, classification head is disabled

## Performance Impact

- **Training**: ~5-10% slower due to additional forward pass
- **Inference**: No impact (classification branch disabled)
- **Memory**: ~50MB additional for classification head parameters

## Dataset Compatibility

Currently configured for:
- GOT10K (with MVTD maritime data)
- Can be extended to LASOT, TrackingNet, etc. by adding annotation files

## Future Improvements

1. Multi-label classification (frame can have multiple challenges)
2. Temporal consistency in classification predictions
3. Use classification predictions to guide tracking
4. Active learning based on classification confidence

## References

- Original HIPTrack paper
- Maritime tracking challenges taxonomy
- See CLASSIFICATION_BRANCH_GUIDE.md for detailed implementation guide

## Contact

For issues or questions, please check the implementation guide or contact the development team.

