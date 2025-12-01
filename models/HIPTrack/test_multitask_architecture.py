#!/usr/bin/env python3
"""
Test script to verify multi-task learning architecture
Run this before starting full training to catch any issues early
"""
import torch
import sys
import os

# Add lib to path
lib_path = os.path.join(os.path.dirname(__file__), 'lib')
sys.path.insert(0, lib_path)

print("=" * 80)
print("Testing HIPTrack Multi-Task Learning Architecture")
print("=" * 80)

# Test 1: Imports
print("\n[Test 1/6] Testing imports...")
try:
    from lib.models.hiptrack.hiptrack_cls import build_hiptrack_cls, HIPTrackCls
    from lib.models.layers.cls_head import ClassificationHead, build_cls_head
    from lib.config.hiptrack.config_cls import cfg, update_config_from_file
    from lib.train.dataset.got10k_cls import Got10kCls
    from lib.train.actors.hiptrack_cls import HIPTrackClsActor
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Config loading
print("\n[Test 2/6] Testing config...")
try:
    yaml_file = os.path.join(os.path.dirname(__file__), 
                             'experiments/hiptrack/hiptrack_cls.yaml')
    update_config_from_file(yaml_file)
    
    assert cfg.MODEL.HIDDEN_DIM == 768, "HIDDEN_DIM should be 768"
    assert cfg.MODEL.CLS_HEAD.NUM_CLASSES == 10, "NUM_CLASSES should be 10"
    assert cfg.MODEL.CLS_HEAD.HIDDEN_DIM == 512, "CLS hidden dim should be 512"
    assert not hasattr(cfg.MODEL, 'BOTTLENECK_DIM') or cfg.MODEL.BOTTLENECK_DIM == cfg.MODEL.HIDDEN_DIM, \
        "BOTTLENECK_DIM should be removed or equal to HIDDEN_DIM"
    
    print("✅ Config loaded successfully")
    print(f"   - Backbone dim: {cfg.MODEL.HIDDEN_DIM}")
    print(f"   - Cls hidden dim: {cfg.MODEL.CLS_HEAD.HIDDEN_DIM}")
    print(f"   - Num classes: {cfg.MODEL.CLS_HEAD.NUM_CLASSES}")
    print(f"   - Cls weight: {cfg.TRAIN.CLS_WEIGHT}")
except Exception as e:
    print(f"❌ Config test failed: {e}")
    sys.exit(1)

# Test 3: Classification head
print("\n[Test 3/6] Testing classification head...")
try:
    cls_head = build_cls_head(cfg)
    
    # Test forward pass
    batch_size = 4
    features = torch.randn(batch_size, 768)
    
    cls_logits = cls_head(features)
    
    # Verify output shape
    assert cls_logits.shape == (batch_size, 10), \
        f"Expected shape (4, 10), got {cls_logits.shape}"
    
    # Verify output is just logits (not dict)
    assert isinstance(cls_logits, torch.Tensor), \
        "Output should be tensor, not dict"
    
    print("✅ Classification head works correctly")
    print(f"   - Input shape: {features.shape}")
    print(f"   - Output shape: {cls_logits.shape}")
    print(f"   - Parameters: {sum(p.numel() for p in cls_head.parameters())/1e3:.1f}K")
except Exception as e:
    print(f"❌ Classification head test failed: {e}")
    sys.exit(1)

# Test 4: Model building
print("\n[Test 4/6] Testing model building...")
try:
    # Temporarily modify config to avoid loading pretrained weights
    original_pretrain = cfg.MODEL.PRETRAIN_FILE
    cfg.MODEL.PRETRAIN_FILE = "mae_pretrain_vit_base.pth"
    
    model = build_hiptrack_cls(cfg, training=True)
    
    # Restore original
    cfg.MODEL.PRETRAIN_FILE = original_pretrain
    
    # Check model has cls_head
    assert hasattr(model, 'cls_head'), "Model should have cls_head attribute"
    assert hasattr(model, 'use_cls_branch'), "Model should have use_cls_branch attribute"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    cls_params = sum(p.numel() for p in model.cls_head.parameters())
    
    print("✅ Model built successfully")
    print(f"   - Total parameters: {total_params/1e6:.2f}M")
    print(f"   - Classification head parameters: {cls_params/1e3:.1f}K")
    print(f"   - Cls branch enabled: {model.use_cls_branch}")
except Exception as e:
    print(f"❌ Model building failed: {e}")
    print("   Note: This might fail if pretrained weights are not available.")
    print("   You can ignore this if you plan to train from scratch.")

# Test 5: Forward pass
print("\n[Test 5/6] Testing forward pass...")
try:
    model = build_hiptrack_cls(cfg, training=True)
    model.eval()
    
    # Create dummy inputs
    batch_size = 2
    template = torch.randn(batch_size, 3, 192, 192)
    search = [torch.randn(batch_size, 3, 384, 384) for _ in range(5)]
    template_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]] * batch_size).unsqueeze(0)
    cls_labels = torch.randint(0, 10, (5, batch_size))  # [5 frames, batch]
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            template=template,
            search=search,
            template_boxes=template_boxes,
            cls_labels=cls_labels
        )
    
    # Verify outputs
    assert isinstance(outputs, list), "Output should be list"
    assert len(outputs) == 5, f"Should have 5 outputs, got {len(outputs)}"
    
    for i, out in enumerate(outputs):
        assert 'pred_boxes' in out, f"Output {i} missing pred_boxes"
        assert 'cls_logits' in out, f"Output {i} missing cls_logits"
        assert out['pred_boxes'].shape[0] == batch_size, "Batch size mismatch in pred_boxes"
        assert out['cls_logits'].shape == (batch_size, 10), \
            f"Expected cls_logits shape ({batch_size}, 10), got {out['cls_logits'].shape}"
    
    print("✅ Forward pass successful")
    print(f"   - Number of outputs: {len(outputs)}")
    print(f"   - Bbox shape: {outputs[0]['pred_boxes'].shape}")
    print(f"   - Cls logits shape: {outputs[0]['cls_logits'].shape}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Inference mode (cls disabled)
print("\n[Test 6/6] Testing inference mode...")
try:
    model.disable_cls_branch()
    
    with torch.no_grad():
        out = model.forward_track(
            index=11,  # Use index > 10 to trigger normal tracking
            template=template,
            template_boxes=template_boxes[0],
            search=search[0]
        )
    
    assert 'pred_boxes' in out, "Output should have pred_boxes"
    # Classification should not be computed in inference
    assert 'cls_logits' not in out or out['cls_logits'] is None, \
        "cls_logits should not be computed during inference"
    
    print("✅ Inference mode works correctly")
    print("   - Classification branch disabled: ✓")
    print("   - Tracking still works: ✓")
except Exception as e:
    print(f"❌ Inference mode test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("✅ All tests passed!")
print("=" * 80)
print("\nYour multi-task learning architecture is ready to train!")
print("\nNext steps:")
print("1. Verify your data annotations:")
print("   python tracking/test_cls_annotations.py")
print("\n2. Start training:")
print("   python tracking/train.py --script hiptrack --config hiptrack_cls --save_dir ./output --mode single")
print("\n3. Monitor training:")
print("   tensorboard --logdir ./tensorboard/train/hiptrack/hiptrack_cls")
print("=" * 80)

