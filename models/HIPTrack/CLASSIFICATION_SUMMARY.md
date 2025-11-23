# HIPTrack Classification Branch - Quick Summary

## What Was Implemented

Added auxiliary classification branch to HIPTrack to classify 10 maritime tracking challenges during training.

## Files Created (10 new + 1 modified)

### Core Files
1. `lib/models/layers/cls_head.py` - Classification head
2. `lib/models/hiptrack/hiptrack_cls.py` - Model with classification
3. `lib/utils/cls_loss.py` - Classification losses
4. `lib/config/hiptrack/config_cls.py` - Config
5. `lib/train/dataset/got10k_cls.py` - Dataset with cls labels
6. `lib/train/actors/hiptrack_cls.py` - Training actor
7. `lib/train/train_script_cls.py` - Training script
8. `lib/test/tracker/hiptrack_cls.py` - Test tracker
9. `lib/test/parameter/hiptrack_cls.py` - Test parameters
10. `experiments/hiptrack/hiptrack_cls.yaml` - YAML config

### Helper Files
- `tracking/convert_cls_to_tracking.py` - Convert checkpoint
- `tracking/test_cls_annotations.py` - Test annotations
- `CLASSIFICATION_BRANCH_USAGE.md` - Full usage guide

### Modified Files
- `lib/train/data/sampler.py` - Added cls_labels handling

## 10 Classification Classes

1. Occlusion (0)
2. Illumination Change (1)
3. Scale Variation (2)
4. Motion Blur (3)
5. Variance in Appearance (4)
6. Partial Visibility (5)
7. Low Resolution (6)
8. Background Clutter (7)
9. Low Contrast Object (8)
10. Normal (9)

### Label Selection Logic
- **Priority 1**: VLM response (use first flagged challenge)
- **Priority 2**: CV response (if multiple flagged, pick randomly)
- **Default**: Normal if no challenges detected

## Quick Start

### Training
```bash
cd /home/thinhnp/MOT/models/HIPTrack
python tracking/train.py --script hiptrack --config hiptrack_cls --save_dir ./output --mode single
```

### Testing (classification auto-disabled)
```bash
python tracking/test.py hiptrack_cls hiptrack_cls --dataset got10k_test --threads 4
```

### Test Annotations Loading
```bash
python tracking/test_cls_annotations.py
```

## Key Features

- **Modular**: Classification branch is independent
- **Training**: Used to improve feature learning
- **Inference**: Automatically disabled during testing
- **Flexible**: Supports CE, Focal, and Label Smoothing losses
- **No Impact**: Zero impact on inference speed

## Configuration

Main config: `experiments/hiptrack/hiptrack_cls.yaml`

Key parameters:
- `CLS_WEIGHT: 1.0` - Classification loss weight
- `CLS_LOSS_TYPE: "CE"` - Loss type
- `NUM_CLASSES: 10` - Number of classes
- `CLS_ANN_DIR` - Annotation directory

## Annotation Format

Location: `/home/thinhnp/MOT/data/train_maritime_env_clf_annts/{seq_name}/{seq_name}.jsonl`

Format: JSONL with frame-level annotations including:
- `cv_response`: Computer vision features (scale, resolution, contrast)
- `vlm_response`: Vision-language model predictions (occlusion, blur, etc.)

## Architecture

```
Template Image ──┐
                 ├─> ViT Backbone ──┬─> Box Head ─> Tracking Predictions
Search Images ───┘                  │
                                    └─> Classification Head ─> Challenge Class
                                        (Only in training)
```

## Logs & Metrics

Training logs include:
- Standard tracking losses (GIoU, L1, Focal)
- Classification loss
- Classification accuracy
- Tracking IoU

## Documentation

- **Full Guide**: `CLASSIFICATION_BRANCH_GUIDE.md`
- **Usage Guide**: `CLASSIFICATION_BRANCH_USAGE.md`
- **This Summary**: `CLASSIFICATION_SUMMARY.md`

## Status

✅ All core components implemented
✅ Classification head with flexible pooling
✅ Multiple loss types (CE, Focal, Label Smoothing)
✅ Dataset integration with maritime annotations
✅ Training and testing scripts
✅ Automatic branch disable in inference
✅ Helper scripts and documentation

## Next Steps

1. Test annotation loading: `python tracking/test_cls_annotations.py`
2. Start training: `python tracking/train.py --script hiptrack --config hiptrack_cls ...`
3. Monitor with TensorBoard: `tensorboard --logdir ./tensorboard/train/hiptrack/hiptrack_cls`
4. Test on benchmark: `python tracking/test.py hiptrack_cls hiptrack_cls ...`

## Notes

- Annotations are in `/home/thinhnp/MOT/data/train_maritime_env_clf_annts/`
- Training data is in `/home/thinhnp/MOT/data/MVTD/train`
- Classification branch is automatically disabled during inference
- Missing annotations use ignore index (-100) and don't contribute to loss

