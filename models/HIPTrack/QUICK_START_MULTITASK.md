# Quick Start: Multi-Task Learning for HIPTrack

## ✨ What Changed?

Your implementation has been **simplified and fixed** to use standard multi-task learning:

### Before (Complex, Broken):
- ❌ Separate classification branch with fusion
- ❌ Dimension mismatch issues (256 vs 768)
- ❌ Classification computed after bbox predictions
- ❌ Fusion features never actually used
- ❌ Complex, hard to debug

### After (Simple, Working):
- ✅ Shared backbone for both tasks
- ✅ Simple classification head
- ✅ Both heads use same features
- ✅ Standard multi-task learning
- ✅ Clean, easy to understand

---

## 🏗️ New Architecture

```
Input Images
     ↓
ViT Backbone (Shared)
     ↓
Features [B, HW, 768]
     ├→ Global Pool → Classification Head → Classes [B, 10]
     └→ Spatial → Box Head → Bboxes [B, 4]
```

**Key Insight**: One backbone learns features for BOTH tasks. The classification task helps the backbone learn more robust features that also benefit bbox prediction.

---

## 📝 Modified Files

✅ **Updated** (5 files):
1. `lib/models/layers/cls_head.py` - Simplified (no fusion layer)
2. `lib/models/hiptrack/hiptrack_cls.py` - Cleaner forward pass
3. `lib/config/hiptrack/config_cls.py` - Removed fusion configs
4. `experiments/hiptrack/hiptrack_cls.yaml` - Updated config
5. `lib/train/actors/hiptrack_cls.py` - Minor fix

📚 **Created** (3 docs):
- `MULTITASK_LEARNING_GUIDE.md` - Complete guide
- `ARCHITECTURE_COMPARISON.md` - Before/after comparison
- `QUICK_START_MULTITASK.md` - This file

🧪 **Created** (1 test):
- `test_multitask_architecture.py` - Verification script

---

## 🚀 Getting Started

### Step 1: Verify Installation

```bash
cd /home/thangdd/workspace/MOT/models/HIPTrack

# Run test script
python test_multitask_architecture.py
```

Expected output:
```
[Test 1/6] Testing imports... ✅
[Test 2/6] Testing config... ✅
[Test 3/6] Testing classification head... ✅
[Test 4/6] Testing model building... ✅
[Test 5/6] Testing forward pass... ✅
[Test 6/6] Testing inference mode... ✅

✅ All tests passed!
```

---

### Step 2: Verify Data Annotations

```bash
# Check if classification annotations are loaded correctly
python tracking/test_cls_annotations.py
```

Should show something like:
```
Loaded classification annotations for X / Y sequences
Class distribution:
  - Occlusion: 234
  - Motion Blur: 456
  - Normal: 1234
  ...
```

---

### Step 3: Start Training

```bash
# Single-stage training
python tracking/train.py \
    --script hiptrack \
    --config hiptrack_cls \
    --save_dir ./output \
    --mode single

# Or distributed training (4 GPUs)
python tracking/train.py \
    --script hiptrack \
    --config hiptrack_cls \
    --save_dir ./output \
    --mode multiple \
    --nproc_per_node 4
```

---

### Step 4: Monitor Training

```bash
# Terminal 1: Training
python tracking/train.py ...

# Terminal 2: TensorBoard
tensorboard --logdir ./tensorboard/train/hiptrack/hiptrack_cls
```

Open browser: http://localhost:6006

**Watch for**:
- `Loss/classification` decreasing
- `Accuracy` increasing (should reach 70-80%)
- `IoU` stable or improving
- `Loss/total` decreasing

---

### Step 5: Test Trained Model

```bash
python tracking/test.py hiptrack_cls hiptrack_cls \
    --dataset got10k_test \
    --threads 4
```

Note: Classification is automatically disabled during testing (no overhead).

---

## ⚙️ Configuration

### Main Config: `experiments/hiptrack/hiptrack_cls.yaml`

```yaml
MODEL:
  HIDDEN_DIM: 768  # Backbone feature dimension (shared)
  CLS_HEAD:
    NUM_CLASSES: 10     # Maritime conditions
    HIDDEN_DIM: 512     # MLP hidden dimension
    DROPOUT: 0.1        # Regularization

TRAIN:
  CLS_WEIGHT: 0.1       # Classification loss weight
  CLS_LOSS_TYPE: "CE"   # Cross-entropy loss
  
  # Two-stage training (optional)
  TWO_STAGE: True
  STAGE1_EPOCHS: 30     # Train cls head only
  STAGE2_EPOCHS: 20     # Fine-tune everything
```

### Tuning Tips

| Parameter | Default | Increase if... | Decrease if... |
|-----------|---------|----------------|----------------|
| `CLS_WEIGHT` | 0.1 | Cls accuracy too low | Tracking degraded |
| `HIDDEN_DIM` | 512 | Need more capacity | Overfitting |
| `DROPOUT` | 0.1 | Overfitting | Underfitting |

---

## 📊 Expected Results

### Training Logs

```
Epoch 1:
  Loss/total: 5.234
  Loss/giou: 0.432
  Loss/l1: 0.234
  Loss/location: 0.123
  Loss/classification: 2.301  ← Start high
  IoU: 0.678
  Accuracy: 0.145  ← Start low (~random)

Epoch 10:
  Loss/classification: 1.234
  Accuracy: 0.456

Epoch 30:
  Loss/classification: 0.543
  Accuracy: 0.789  ← Good performance!
```

### Performance Targets

- **Classification Accuracy**: 70-80% on 10 classes
- **Tracking IoU**: Similar or better than baseline HIPTrack
- **Training Time**: ~same as baseline (minimal overhead)

---

## 🐛 Troubleshooting

### Issue: Test script fails at model building
**Cause**: Pretrained weights not found
**Solution**: Either:
- Download pretrained weights to `pretrained/HipTrack/HIPTrack_got10k.pth.tar`
- Or train from scratch (modify config: `PRETRAIN_FILE: "mae_pretrain_vit_base.pth"`)

### Issue: Classification accuracy stuck at 10%
**Cause**: Random guessing (10 classes)
**Solution**:
1. Check annotations loaded: `python tracking/test_cls_annotations.py`
2. Increase `CLS_WEIGHT` to 0.5
3. Verify labels in dataset

### Issue: Out of memory
**Solution**:
- Reduce `BATCH_SIZE` from 16 to 8
- Or reduce `SEARCH.SIZE` from 384 to 320

### Issue: Tracking performance worse than baseline
**Solution**:
- Reduce `CLS_WEIGHT` to 0.01-0.05
- Classification should act as regularization, not dominate

---

## 📚 Documentation

- **Complete Guide**: `MULTITASK_LEARNING_GUIDE.md`
- **Architecture Comparison**: `ARCHITECTURE_COMPARISON.md`
- **Classification Summary**: `CLASSIFICATION_SUMMARY.md`
- **Usage Guide**: `CLASSIFICATION_BRANCH_USAGE.md`

---

## ✅ Verification Checklist

Before starting training, verify:

- [ ] Test script passes all 6 tests
- [ ] Annotations loaded successfully
- [ ] Config file updated (no BOTTLENECK_DIM)
- [ ] Model builds without errors
- [ ] Forward pass returns correct shapes
- [ ] Inference mode disables classification

After training:
- [ ] Classification accuracy > 70%
- [ ] Tracking IoU similar to baseline
- [ ] Model can be loaded for inference
- [ ] Testing disables cls branch automatically

---

## 🎯 Key Advantages

### Simplicity
- **Before**: ~450 lines of complex code
- **After**: ~250 lines of clean code

### Efficiency
- **Before**: Fusion overhead + dimension issues
- **After**: Simple pooling + standard MTL

### Effectiveness
- **Before**: Fusion not working properly
- **After**: Proven multi-task learning approach

### Maintainability
- **Before**: Hard to debug fusion issues
- **After**: Easy to understand and extend

---

## 🔬 How Multi-Task Learning Helps

1. **Feature Learning**: Classification forces backbone to learn semantic features
2. **Regularization**: Auxiliary task prevents overfitting
3. **Robustness**: Shared features are more generalizable
4. **Maritime-Aware**: Model learns to recognize challenging conditions

Example: When model learns to classify "motion blur", it learns features that also help track blurred objects.

---

## 📞 Need Help?

1. **Check logs**: `./output/logs/hiptrack-hiptrack_cls.log`
2. **Read docs**: Start with `MULTITASK_LEARNING_GUIDE.md`
3. **Run test**: `python test_multitask_architecture.py`
4. **Verify data**: `python tracking/test_cls_annotations.py`

---

## 🎉 Summary

Your implementation now uses **clean, standard multi-task learning**:
- ✅ One backbone, two heads (bbox + classification)
- ✅ Simple architecture, proven approach
- ✅ Classification helps tracking through better features
- ✅ No fusion complexity or dimension issues
- ✅ Ready to train!

**Good luck with your training!** 🚀

