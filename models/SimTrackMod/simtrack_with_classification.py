"""
Extended SimTrack Model with Classification Head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import sys

# Add the lib path to import SimTrack modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from lib.models.stark.simtrack import SimTrack
from lib.models.stark.backbone import build_backbone_simtrack
from lib.models.stark.head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh


class SimTrackWithClassification(nn.Module):
    """
    Extended SimTrack with Classification Head
    Architecture: backbone -> [CLS] token -> linear(512) -> classifier(10 classes)
    Dual stream: one for classification loss, one added to traditional prediction
    """
    def __init__(self, backbone, box_head, num_classes=10, hidden_dim=512,
                 aux_loss=False, head_type="CORNER"):
        super().__init__()
        
        # Original SimTrack components
        self.backbone = backbone
        self.box_head = box_head
        self.aux_loss = aux_loss
        self.head_type = head_type
        
        # Bottleneck layer for features
        self.bottleneck = nn.Linear(backbone.num_features, box_head.channel)
        
        if head_type == "CORNER":
            self.feat_sz_s = int(backbone.sz)
            self.feat_len_s = int(backbone.sz ** 2)
        
        # Classification head components
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Linear projection from bottleneck features (box_head.channel) to hidden_dim (512)
        self.cls_projection = nn.Linear(box_head.channel, hidden_dim)
        
        # Classification head (10 classes)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Fusion layer to add classification features to prediction stream
        self.fusion_layer = nn.Linear(hidden_dim, box_head.channel)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, img=None, seq_dict=None, mode="backbone", run_box_head=True, run_cls_head=True):
        if mode == "backbone":
            return self.forward_backbone(img)
        elif mode == "head":
            return self.forward_head(seq_dict, run_box_head=run_box_head, run_cls_head=run_cls_head)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward_backbone(self, input):
        """
        Forward through backbone
        :param input: list [template_img, search_img, template_anno]
        :return: feature dictionary
        """
        # Forward the backbone
        output_back = self.backbone(input)  # features from CLIPVIT
        # Adjust the shapes
        return self.adjust(output_back)

    def forward_head(self, seq_dict, run_box_head=True, run_cls_head=True):
        """
        Forward through head with dual stream architecture
        """
        if self.aux_loss:
            raise ValueError("Deep supervision is not supported.")
        
        # Get features from backbone (seq_dict should contain features)
        if isinstance(seq_dict, list) and len(seq_dict) > 0:
            # Extract CLS token (first token) for classification
            feat = seq_dict[0]['feat']  # Shape: [seq_len, batch, dim]
            cls_token = feat[0]  # CLS token: [batch, dim]
        else:
            feat = seq_dict['feat']
            cls_token = feat[0]  # CLS token: [batch, dim]
        
        outputs = {}
        
        # Classification stream
        if run_cls_head:
            # Project CLS token to hidden dimension
            cls_features = self.cls_projection(cls_token)  # [batch, hidden_dim]
            cls_features = F.relu(cls_features)
            cls_features = self.dropout(cls_features)
            
            # Classification prediction
            cls_logits = self.classifier(cls_features)  # [batch, num_classes]
            outputs['cls_logits'] = cls_logits
            
            # Features for fusion
            fusion_features = self.fusion_layer(cls_features)  # [batch, box_head.channel]
            outputs['cls_features'] = cls_features
        else:
            fusion_features = None
        
        # Box prediction stream
        if run_box_head:
            if isinstance(seq_dict, list):
                output_embed = seq_dict[0]['feat']
            else:
                output_embed = seq_dict['feat']
            
            # Forward the box head with optional fusion
            out, outputs_coord = self.forward_box_head(output_embed, fusion_features)
            outputs.update(out)
            outputs['pred_coords'] = outputs_coord
            outputs['output_embed'] = output_embed
        
        return outputs

    def forward_box_head(self, memory, fusion_features=None):
        """
        Forward through box head with optional classification feature fusion
        """
        # Adjust shape - get search region features
        enc_opt = memory[-self.feat_len_s:].transpose(0, 1)  # [B, HW, C]
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()  # [B, N, C, HW]
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)  # [B*N, C, H, W]
        
        # Fusion with classification features if available
        if fusion_features is not None:
            # Reshape fusion features to match spatial dimensions
            fusion_feat = fusion_features.unsqueeze(-1).unsqueeze(-1)  # [B, C_fusion, 1, 1]
            fusion_feat = fusion_feat.expand(-1, -1, self.feat_sz_s, self.feat_sz_s)  # [B, C_fusion, H, W]
            
            # Repeat for each query
            fusion_feat = fusion_feat.unsqueeze(1).repeat(1, Nq, 1, 1, 1)  # [B, N, C_fusion, H, W]
            fusion_feat = fusion_feat.view(-1, fusion_feat.size(2), self.feat_sz_s, self.feat_sz_s)  # [B*N, C_fusion, H, W]
            
            # Add to box features
            if fusion_feat.size(1) == opt_feat.size(1):
                opt_feat = opt_feat + fusion_feat
            else:
                # If dimensions don't match, use adaptive pooling or projection
                print(f"Warning: Dimension mismatch in fusion. opt_feat: {opt_feat.shape}, fusion_feat: {fusion_feat.shape}")
        
        # Run the corner head
        outputs_coord = box_xyxy_to_cxcywh(self.box_head(opt_feat))
        outputs_coord_new = outputs_coord.view(bs, Nq, 4)
        out = {'pred_boxes': outputs_coord_new}
        
        return out, outputs_coord_new

    def adjust(self, output_back):
        """
        Adjust backbone output features
        """
        src_feat = output_back
        # Reduce channel dimension
        feat = self.bottleneck(src_feat)  # [B, C, H, W] or [seq_len, B, C]
        
        # Adjust shapes for transformer format
        if len(feat.shape) == 4:  # [B, C, H, W]
            feat_vec = feat.flatten(2).permute(1, 0, 2)  # [HW, B, C]
        else:  # Already in [seq_len, B, C] format
            feat_vec = feat.permute(1, 0, 2)  # [HW, B, C]
        
        return {"feat": feat_vec}


def build_simtrack_with_classification(cfg, num_classes=10, hidden_dim=512):
    """
    Build SimTrack model with classification head
    """
    backbone = build_backbone_simtrack(cfg)
    box_head = build_box_head(cfg)
    
    model = SimTrackWithClassification(
        backbone=backbone,
        box_head=box_head,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        aux_loss=cfg.TRAIN.DEEP_SUPERVISION if hasattr(cfg.TRAIN, 'DEEP_SUPERVISION') else False,
        head_type=cfg.MODEL.HEAD_TYPE if hasattr(cfg.MODEL, 'HEAD_TYPE') else "CORNER"
    )
    
    return model


def load_pretrained_weights(model, weight_path, strict=False):
    """
    Load pretrained weights into the model
    """
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    
    # Load weights
    checkpoint = torch.load(weight_path, map_location='cpu')
    
    # Get model state dict
    model_dict = model.state_dict()
    
    # Filter checkpoint to match model keys
    filtered_dict = {}
    for k, v in checkpoint.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                filtered_dict[k] = v
            else:
                print(f"Shape mismatch for {k}: model {model_dict[k].shape} vs checkpoint {v.shape}")
        else:
            print(f"Key {k} not found in model")
    
    # Load filtered weights
    missing_keys, unexpected_keys = model.load_state_dict(filtered_dict, strict=strict)
    
    print(f"Loaded {len(filtered_dict)} parameters from {weight_path}")
    if missing_keys:
        print(f"Missing keys: {len(missing_keys)} (these will be randomly initialized)")
        print("First 5 missing keys:", missing_keys[:5])
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)}")
    
    return model


def freeze_backbone(model):
    """
    Freeze all backbone parameters to prevent training
    """
    frozen_params = 0
    total_params = 0
    
    # Freeze backbone
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False
        frozen_params += param.numel()
        print(f"Frozen: {name}")
    
    # Freeze bottleneck (connection between backbone and head)
    for name, param in model.bottleneck.named_parameters():
        param.requires_grad = False
        frozen_params += param.numel()
        print(f"Frozen: bottleneck.{name}")
    
    # Count total parameters
    for param in model.parameters():
        total_params += param.numel()
    
    trainable_params = total_params - frozen_params
    
    print(f"\nFreeze Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Frozen parameters: {frozen_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen ratio: {frozen_params/total_params*100:.1f}%")
    
    return model


def unfreeze_backbone(model):
    """
    Unfreeze all backbone parameters to allow training
    """
    unfrozen_params = 0
    
    # Unfreeze backbone
    for name, param in model.backbone.named_parameters():
        param.requires_grad = True
        unfrozen_params += param.numel()
        
    # Unfreeze bottleneck
    for name, param in model.bottleneck.named_parameters():
        param.requires_grad = True
        unfrozen_params += param.numel()
    
    print(f"Unfrozen {unfrozen_params:,} backbone parameters")
    return model


class DummyConfig:
    """Dummy config for testing"""
    def __init__(self):
        self.MODEL = DummyModelConfig()
        self.TRAIN = DummyTrainConfig()
        self.DATA = DummyDataConfig()

class DummyModelConfig:
    def __init__(self):
        self.BACKBONE = DummyBackboneConfig()
        self.HEAD_TYPE = "CORNER"
        self.HIDDEN_DIM = 256  # Changed to match bottleneck output
        self.HEAD_DIM = 256

class DummyBackboneConfig:
    def __init__(self):
        self.TYPE = "ViT-B/16"
        self.NUM_FEA = 768
        self.TEMPLATE_SZ = 112
        self.SEARCH_SZ = 224
        self.WINDOW_SZ = 16
        self.FOVEAL_SZ = 64
        self.DILATION = False

class DummyTrainConfig:
    def __init__(self):
        self.DEEP_SUPERVISION = False

class DummyDataConfig:
    def __init__(self):
        self.SEARCH = DummySearchConfig()

class DummySearchConfig:
    def __init__(self):
        self.SIZE = 224


def unit_test():
    """
    Unit test with dummy data: 1 template image, 1 search frame, 1 class ID
    """
    print("Starting unit test...")
    
    # Create dummy config
    cfg = DummyConfig()
    
    # Build model
    print("Building model...")
    model = build_simtrack_with_classification(cfg, num_classes=10, hidden_dim=512)
    
    # Load pretrained weights
    weight_path = "./SimTrackMod/checkpoints/sim-vit-b-16.pth"
    print(f"Loading weights from {weight_path}...")
    model = load_pretrained_weights(model, weight_path, strict=False)
    
    # Force model to CPU to avoid device mismatch
    model = model.cpu()
    print("Model moved to CPU")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input data (ensure all on CPU)
    batch_size = 1
    template_size = 112
    search_size = 224
    
    # Dummy images (random data for testing) - explicitly on CPU
    template_img = torch.randn(batch_size, 3, template_size, template_size, device='cpu')
    search_img = torch.randn(batch_size, 3, search_size, search_size, device='cpu')
    
    # Dummy annotation (x, y, w, h) normalized [0, 1]
    template_anno = torch.tensor([[0.4, 0.4, 0.2, 0.2]], dtype=torch.float32, device='cpu')  # center annotation
    
    # Dummy class ID
    target_class = torch.tensor([0], dtype=torch.long, device='cpu')  # class 0
    
    print(f"Input shapes:")
    print(f"  Template: {template_img.shape}")
    print(f"  Search: {search_img.shape}")
    print(f"  Annotation: {template_anno.shape}")
    print(f"  Target class: {target_class.shape}")
    
    with torch.no_grad():
        # Forward backbone
        print("\nForward backbone...")
        input_data = [template_img, search_img, template_anno]
        backbone_output = model.forward_backbone(input_data)
        
        print(f"Backbone output keys: {backbone_output.keys()}")
        print(f"Feature shape: {backbone_output['feat'].shape}")
        
        # Forward head (both classification and box prediction)
        print("\nForward head...")
        seq_dict = [backbone_output]  # Match the expected format
        
        head_output = model.forward_head(seq_dict, run_box_head=True, run_cls_head=True)
        
        print(f"Head output keys: {head_output.keys()}")
        
        if 'cls_logits' in head_output:
            cls_logits = head_output['cls_logits']
            print(f"Classification logits shape: {cls_logits.shape}")
            print(f"Predicted class probabilities: {F.softmax(cls_logits, dim=1)}")
            predicted_class = torch.argmax(cls_logits, dim=1)
            print(f"Predicted class: {predicted_class.item()}")
        
        if 'pred_boxes' in head_output:
            pred_boxes = head_output['pred_boxes']
            print(f"Predicted boxes shape: {pred_boxes.shape}")
            print(f"Predicted boxes (cxcywh): {pred_boxes}")
        
        if 'cls_features' in head_output:
            cls_features = head_output['cls_features']
            print(f"Classification features shape: {cls_features.shape}")
    
    print("\nUnit test completed successfully!")
    return model, head_output


if __name__ == "__main__":
    # Run unit test
    model, outputs = unit_test()