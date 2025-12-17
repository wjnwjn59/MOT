# hiptrack_cls.py
"""
HIPTrack with an additional classification head for maritime condition / task labels.
Multi-task: same backbone and bbox path, just an extra head on the bottleneck feature.
"""

import os
import torch
from torch import nn

from lib.models.hiptrack.hiptrack import HIPTrack
from lib.models.hiptrack.vit import vit_base_patch16_224
from lib.models.hiptrack.vit_ce import (
    vit_large_patch16_224_ce,
    vit_base_patch16_224_ce
)
from lib.models.layers.head import build_box_head
from lib.models.layers.cls_head import build_cls_head  # you already had this in your previous design
from lib.utils.box_ops import box_xyxy_to_cxcywh


class HIPTrackCls(HIPTrack):
    """
    HIPTrack with an additional classification head on top of the bbox bottleneck feature.

    - BBox branch is unchanged (same computation, same box_head).
    - Classification head sees the same fused search feature (after searchRegionFusion).
    - This is classic multi-task learning: shared backbone, 2 heads.
    """

    def __init__(self,
                 transformer,
                 box_head,
                 cls_head,
                 aux_loss=False,
                 head_type="CORNER",
                 vis_during_train=False,
                 new_hip=False,
                 memory_max=150,
                 update_interval=20,
                 use_cls_head=True):
        super().__init__(
            transformer=transformer,
            box_head=box_head,
            aux_loss=aux_loss,
            head_type=head_type,
            vis_during_train=vis_during_train,
            new_hip=new_hip,
            memory_max=memory_max,
            update_interval=update_interval,
        )

        # Extra classification head (e.g. maritime condition / task head)
        self.cls_head = cls_head
        self.use_cls_head = use_cls_head

    def enable_cls_head(self):
        self.use_cls_head = True
        for p in self.cls_head.parameters():
            p.requires_grad = True

    def disable_cls_head(self):
        self.use_cls_head = False
        for p in self.cls_head.parameters():
            p.requires_grad = False

    def forward_head(self, cat_feature, gt_score_map=None, return_topk_boxes=False):
        """
        Same as HIPTrack.forward_head, plus:
        - global pooling on the fused search feature
        - classification head on top of that pooled feature
        - classification logits attached into the output dict as 'cls_logits'

        IMPORTANT:
        - BBox path and box_head usage are unchanged.
        - No fusion back into detection features.
        """
        # cat_feature: [2, B, HW, C]
        _, B, HW, C = cat_feature.shape
        H = int(HW ** 0.5)
        W = H

        # [B, H, W, C] -> [B, C, H, W]
        originSearch = cat_feature[0].view(B, H, W, C).permute(0, 3, 1, 2)
        dynamicSearch = cat_feature[1].view(B, H, W, C).permute(0, 3, 1, 2)

        # This is the same bottleneck feature used by the original HIPTrack
        fused_search = self.searchRegionFusion(originSearch + dynamicSearch)  # [B, C, H, W]

        # -------- Classification head (multi-task) --------
        cls_logits = None
        if self.use_cls_head and self.training:
            # global average pooling
            # shape: [B, C]
            global_feat = fused_search.mean(dim=[2, 3])
            # cls_head is expected to take [B, C] -> [B, num_classes]
            cls_logits = self.cls_head(global_feat)

        # -------- Original bbox head path (unchanged) --------
        enc_opt = fused_search.view(B, C, HW).permute(0, 2, 1)  # [B, HW, C]
        opt = enc_opt.unsqueeze(-1).permute(0, 3, 2, 1).contiguous()  # [B, 1, C, HW]
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)  # [B, C, H, W]

        if self.head_type == "CORNER":
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {
                'pred_boxes': outputs_coord_new,
                'score_map': score_map,
            }
        elif self.head_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map, topkBbox = self.box_head(
                opt_feat, gt_score_map, return_topk_boxes
            )
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            if return_topk_boxes:
                out = {
                    'pred_boxes': outputs_coord_new,
                    'score_map': score_map_ctr,
                    'size_map': size_map,
                    'offset_map': offset_map,
                    'topk_pred_boxes': topkBbox,
                }
            else:
                out = {
                    'pred_boxes': outputs_coord_new,
                    'score_map': score_map_ctr,
                    'size_map': size_map,
                    'offset_map': offset_map,
                }
        else:
            raise NotImplementedError

        # Attach classification output if computed
        if cls_logits is not None:
            out['cls_logits'] = cls_logits

        return out


def build_hiptrack_cls(cfg, training=True):
    """
    Build HIPTrackCls (HIPTrack + classification head) for multi-task learning.

    - Backbone, HIP and bbox head are identical to original HIPTrack.
    - cls_head is an extra head on top of the same bottleneck feature.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_root = os.path.join(current_dir, '../../..')

    # Pretrained backbone path
    if cfg.MODEL.PRETRAIN_FILE and ('HIPTrack' not in cfg.MODEL.PRETRAIN_FILE
                                    and 'DropTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_root, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    # ---------- Backbone ----------
    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(
            pretrained,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE
        )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(
            pretrained,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
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
        raise NotImplementedError(
            f"Backbone {cfg.MODEL.BACKBONE.TYPE} not supported in HIPTrackCls"
        )

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    # ---------- BBox head (same as original HIPTrack) ----------
    box_head = build_box_head(cfg, hidden_dim)

    # ---------- Classification head ----------
    # Assumes build_cls_head(cfg) returns a module taking [B, hidden_dim] -> [B, num_classes]
    cls_head = build_cls_head(cfg)

    # ---------- Model ----------
    model = HIPTrackCls(
        transformer=backbone,
        box_head=box_head,
        cls_head=cls_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        new_hip=cfg.MODEL.NEW_HIP,
        memory_max=cfg.MODEL.MAX_MEM,
        update_interval=cfg.TEST.UPDATE_INTERVAL,
        use_cls_head=training,  # enable only during training by default
    )

    # ---------- Load pretrained HIPTrack weights if provided ----------
    if ('HIPTrack' in cfg.MODEL.PRETRAIN_FILE or 'DropTrack' in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained_path = os.path.join(pretrained_root, cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["net"], strict=False
        )
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        print('Missing keys (expected for new cls_head):', missing_keys)
        print('Unexpected keys:', unexpected_keys)

    return model
