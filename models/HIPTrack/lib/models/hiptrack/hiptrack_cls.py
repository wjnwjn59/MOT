"""
HIPTrack model with auxiliary classification head for multi-task learning
"""
import torch
from torch import nn
from lib.models.hiptrack.hiptrack import HIPTrack
from lib.models.layers.cls_head import build_cls_head
import os


class HIPTrackCls(HIPTrack):
    """
    HIPTrack with auxiliary classification head for multi-task learning.
    Both tasks (tracking and classification) share the same backbone features.
    Classification branch is only used during training.
    """
    
    def __init__(self, transformer, box_head, cls_head, 
                 aux_loss=False, head_type="CORNER", 
                 vis_during_train=False, new_hip=False, 
                 memory_max=150, update_interval=20,
                 use_cls_branch=True):
        """
        Args:
            transformer: Backbone transformer
            box_head: Bounding box prediction head
            cls_head: Classification head (auxiliary)
            use_cls_branch: Whether to use classification branch
        """
        super().__init__(
            transformer=transformer,
            box_head=box_head,
            aux_loss=aux_loss,
            head_type=head_type,
            vis_during_train=vis_during_train,
            new_hip=new_hip,
            memory_max=memory_max,
            update_interval=update_interval
        )
        
        # Classification head
        self.cls_head = cls_head
        self.use_cls_branch = use_cls_branch
    
    def enable_cls_branch(self):
        """Enable classification branch (for training)"""
        self.use_cls_branch = True
        for param in self.cls_head.parameters():
            param.requires_grad = True
    
    def disable_cls_branch(self):
        """Disable classification branch (for inference)"""
        self.use_cls_branch = False
        for param in self.cls_head.parameters():
            param.requires_grad = False
    
    def forward_classification(self, features):
        """
        Forward pass for classification branch
        
        Args:
            features: Pooled features from search region [B, C]
        Returns:
            cls_logits: Classification logits [B, num_classes]
        """
        if not self.use_cls_branch:
            return None
        
        return self.cls_head(features)
    
    def forward(self, template: torch.Tensor,
                search: list,
                search_after: torch.Tensor=None,
                previous: torch.Tensor=None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                gtBoxes=None,
                previousBoxes=None,
                template_boxes=None,
                cls_labels=None
                ):
        """
        Forward pass with multi-task learning (tracking + classification)
        
        Args:
            template: Template image [B, 3, H_t, W_t]
            search: List of search images [5 x [B, 3, H_s, W_s]]
            cls_labels: Classification labels [5, B] (one per search frame)
            ... (other args same as parent)
        
        Returns:
            outputs: List of dicts, each containing:
                - pred_boxes: Bounding box predictions
                - score_map: Score maps
                - cls_logits: Classification predictions (if use_cls_branch)
                - backbone_feat: Backbone features
        """
        # Call parent forward for tracking predictions
        outputs = super().forward(
            template=template,
            search=search,
            search_after=search_after,
            previous=previous,
            ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
            return_last_attn=return_last_attn,
            gtBoxes=gtBoxes,
            previousBoxes=previousBoxes,
            template_boxes=template_boxes
        )
        
        # Add classification predictions if enabled
        if self.use_cls_branch and cls_labels is not None:
            B, _, Ht, Wt = template.shape
            num_template_patches = (Ht // 16) ** 2
            
            for out in outputs:
                # Extract backbone features (already computed by parent)
                backbone_feat = out['backbone_feat']  # [B, HW_template+HW_search, C]
                
                # Extract search region features only (skip template patches)
                search_feat = backbone_feat[:, num_template_patches:, :]  # [B, HW_search, C]
                
                # Global pooling: average over spatial dimensions
                # This creates a single feature vector per sample
                global_feat = search_feat.mean(dim=1)  # [B, C]
                
                # Get classification prediction using the same features as bbox
                cls_logits = self.forward_classification(global_feat)
                
                # Add to output dict
                if cls_logits is not None:
                    out['cls_logits'] = cls_logits
        
        return outputs
    
    def forward_track(self, index: int, template: torch.Tensor, 
                     template_boxes: torch.Tensor, search: torch.Tensor, 
                     ce_template_mask=None, ce_keep_rate=None, 
                     searchRegionImg=None, info=None):
        """
        Forward pass for tracking (inference)
        Classification branch is disabled during inference
        """
        # Temporarily disable classification
        was_enabled = self.use_cls_branch
        self.disable_cls_branch()
        
        # Call parent tracking
        out = super().forward_track(
            index=index,
            template=template,
            template_boxes=template_boxes,
            search=search,
            ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
            searchRegionImg=searchRegionImg,
            info=info
        )
        
        # Restore classification state
        if was_enabled:
            self.enable_cls_branch()
        
        return out


def build_hiptrack_cls(cfg, training=True):
    """
    Build HIPTrack model with classification head
    """
    from lib.models.hiptrack.vit import vit_base_patch16_224
    from lib.models.hiptrack.vit_ce import (
        vit_large_patch16_224_ce, 
        vit_base_patch16_224_ce
    )
    from lib.models.layers.head import build_box_head
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, '../../..')
    
    # Build backbone
    if cfg.MODEL.PRETRAIN_FILE and ('HIPTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''
    
    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(
            pretrained, 
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
        )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(
            pretrained, 
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE
        )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(
            pretrained, 
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
        )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    else:
        raise NotImplementedError(f"Backbone {cfg.MODEL.BACKBONE.TYPE} not supported")
    
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
    
    # Build box head
    box_head = build_box_head(cfg, hidden_dim)
    
    # Build classification head (shares same feature dimension as backbone)
    cls_head = build_cls_head(cfg)
    
    # Build model
    model = HIPTrackCls(
        transformer=backbone,
        box_head=box_head,
        cls_head=cls_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        new_hip=cfg.MODEL.NEW_HIP,
        memory_max=cfg.MODEL.MAX_MEM,
        update_interval=cfg.TEST.UPDATE_INTERVAL,
        use_cls_branch=training  # Enable cls branch only during training
    )
    
    # Load pretrained weights if specified
    if 'HIPTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        pretrained_full_path = os.path.join(
            current_dir, '../../..', 
            cfg.MODEL.PRETRAIN_FILE
        )
        checkpoint = torch.load(pretrained_full_path, map_location="cpu")
        # Load only matching keys (tracking part)
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["net"], strict=False
        )
        print(f'Load pretrained model from: {cfg.MODEL.PRETRAIN_FILE}')
        print(f'Missing keys: {missing_keys}')
        print(f'Unexpected keys: {unexpected_keys}')
    
    return model
