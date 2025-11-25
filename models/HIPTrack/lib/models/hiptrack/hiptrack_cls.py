"""
HIPTrack model with auxiliary classification branch
"""
import torch
from torch import nn
from lib.models.hiptrack.hiptrack import HIPTrack
from lib.models.layers.cls_head import build_cls_head
import os


class HIPTrackCls(HIPTrack):
    """
    HIPTrack with auxiliary classification branch
    Classification branch is only used during training
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
    
    def forward_classification(self, cls_token):
        """
        Forward pass for classification branch using CLS token
        
        Args:
            cls_token: CLS token from ViT backbone [B, C]
        Returns:
            dict: {
                'cls_logits': [B, num_classes],
                'cls_features': [B, hidden_dim],
                'fusion_features': [B, bottleneck_dim]
            }
        """
        if not self.use_cls_branch:
            return None
        
        # Classification head returns dict with logits, features, and fusion
        cls_output = self.cls_head(cls_token)
        return cls_output
    
    def _get_template_search_indices(self, Ht, Wt, has_cls_token=True):
        """
        Get correct indices for template and search features
        
        Args:
            Ht, Wt: Template height and width
            has_cls_token: Whether CLS token is present
        
        Returns:
            tuple: (template_start, template_end, search_start)
        """
        num_template_patches = (Ht // 16) ** 2
        
        if has_cls_token:
            # Format: [CLS, template, search]
            template_start = 1
            template_end = 1 + num_template_patches
            search_start = 1 + num_template_patches
        else:
            # Format: [template, search]
            template_start = 0
            template_end = num_template_patches
            search_start = num_template_patches
        
        return template_start, template_end, search_start
    
    def _extract_features_with_cls(self, x, Ht, Wt):
        """
        Extract template and search features when CLS token is present
        
        Args:
            x: Backbone output [B, 1+HW_t+HW_s, C] with CLS token at position 0
            Ht, Wt: Template height and width
        
        Returns:
            tuple: (template_features, search_features) without CLS token
        """
        # When add_cls_token=True, sequence is: [CLS, template_patches, search_patches]
        # CLS token at index 0
        # Template: indices 1 to 1+(Ht//16)^2
        # Search: indices 1+(Ht//16)^2 to end
        
        template_start, template_end, search_start = self._get_template_search_indices(Ht, Wt, has_cls_token=True)
        
        # Extract without CLS token
        template_feat = x[:, template_start:template_end, :]  # [B, HW_t, C]
        search_feat = x[:, search_start:, :]                   # [B, HW_s, C]
        
        return template_feat, search_feat
    
    def _extract_cls_and_fuse(self, backbone_feat, cat_feature):
        """
        Extract CLS token, get classification output, and apply fusion
        
        Args:
            backbone_feat: Full backbone output [B, 1+HW_t+HW_s, C] with CLS token at pos 0
            cat_feature: Search features for box head [2, B, HW, C]
        
        Returns:
            tuple: (cls_output_dict, fused_cat_feature)
        """
        if not self.use_cls_branch:
            return None, cat_feature
        
        # Extract CLS token at position 0
        cls_token = backbone_feat[:, 0, :]  # [B, C]
        
        # Get classification output with fusion features
        cls_output = self.forward_classification(cls_token)
        
        # Apply fusion to cat_feature if available
        if cls_output is not None and 'fusion_features' in cls_output:
            fusion_feat = cls_output['fusion_features']  # [B, bottleneck_dim]
            
            # Add fusion as residual to both original and dynamic features
            # cat_feature: [2, B, HW, C]
            _, B, HW, C = cat_feature.shape
            
            # Expand fusion to match spatial dimensions
            fusion_expanded = fusion_feat.unsqueeze(1).expand(-1, HW, -1)  # [B, HW, C]
            
            # Add residual to both streams if dimensions match
            if fusion_expanded.shape[-1] == cat_feature.shape[-1]:
                fused_cat_feature = cat_feature + fusion_expanded.unsqueeze(0)
            else:
                fused_cat_feature = cat_feature
        else:
            fused_cat_feature = cat_feature
        
        return cls_output, fused_cat_feature
    
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
                cls_labels=None  # NEW: classification labels
                ):
        """
        Forward pass with classification (SimTrackMod architecture)
        
        Additional Args:
            cls_labels: Classification labels [num_search_frames, batch]
        """
        # Call parent forward for tracking
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
            
            for out in outputs:
                # Extract search features from backbone output
                # backbone_feat shape: [B, HW_template+HW_search, C] (no CLS token for now)
                search_feat = out['backbone_feat']
                
                # Extract search region features (skip template patches)
                num_template_patches = (Ht // 16) ** 2
                search_only_feat = search_feat[:, num_template_patches:, :]  # [B, HW_search, C]
                
                # Create global feature via mean pooling (simulates CLS token)
                # SimTrackMod uses CLS token, we approximate with mean pooling
                global_feat = search_only_feat.mean(dim=1)  # [B, C]
                
                # Get classification prediction
                cls_output = self.forward_classification(global_feat)
                
                # Add to output dict
                if cls_output is not None:
                    out['cls_logits'] = cls_output['cls_logits']
                    out['cls_features'] = cls_output['cls_features']
                    out['fusion_features'] = cls_output['fusion_features']
            
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

    def forward_head(self, cat_feature, gt_score_map=None, return_topk_boxes=False, fusion_features=None):
        """
        Forward head with optional fusion from classification branch
        
        Args:
            cat_feature: [2, B, HW, C] - original and dynamic search features
            fusion_features: [B, C'] - fusion features from classification branch
        """
        from lib.utils.box_ops import box_xyxy_to_cxcywh
        
        _, B, HW, C = cat_feature.shape
        H = int(HW ** 0.5)
        W = H
        
        originSearch = cat_feature[0].view(B, H, W, C).permute(0, 3, 1, 2)
        dynamicSearch = cat_feature[1].view(B, H, W, C).permute(0, 3, 1, 2)
        
        # Fuse original and dynamic search features
        fused_search = self.searchRegionFusion(originSearch + dynamicSearch)  # [B, C, H, W]
        
        # If classification fusion features available, apply residual connection
        if fusion_features is not None and self.use_cls_branch:
            # Expand fusion features to spatial dimensions and add as residual
            # fusion_features: [B, C'] where C' should match C (bottleneck_dim)
            fusion_spatial = fusion_features.unsqueeze(-1).unsqueeze(-1)  # [B, C', 1, 1]
            fusion_spatial = fusion_spatial.expand(-1, -1, H, W)  # [B, C', H, W]
            
            # Residual connection (SimTrackMod style)
            # Assuming C' = C (both are bottleneck_dim = 256)
            if fusion_spatial.shape[1] == fused_search.shape[1]:
                fused_search = fused_search + fusion_spatial
        
        enc_opt = fused_search.view(B, C, HW).permute(0, 2, 1)  # [B, HW, C]
        
        # Prepare for box head
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map, topkBbox = self.box_head(opt_feat, gt_score_map, return_topk_boxes)
            outputs_coord = bbox 
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            if return_topk_boxes:
                out = {'pred_boxes': outputs_coord_new,
                       'score_map': score_map_ctr,
                       'size_map': size_map,
                       'offset_map': offset_map,
                       'topk_pred_boxes': topkBbox,
                    }
            else:
                out = {'pred_boxes': outputs_coord_new,
                       'score_map': score_map_ctr,
                       'size_map': size_map,
                       'offset_map': offset_map,
                    }
            return out
        else:
            raise NotImplementedError


def build_hiptrack_cls(cfg, training=True):
    """
    Build HIPTrack model with classification branch
    """
    from lib.models.hiptrack.vit import vit_base_patch16_224
    from lib.models.hiptrack.vit_ce import (
        vit_large_patch16_224_ce, 
        vit_base_patch16_224_ce
    )
    from lib.models.layers.head import build_box_head
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    
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
    
    # Build classification head
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
            current_dir, '../../../pretrained_models', 
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

