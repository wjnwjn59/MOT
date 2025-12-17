from lib.train.actors.hiptrack import HIPTrackActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.heapmap_utils import generate_heatmap


class HIPTrackClsActor(HIPTrackActor):
    """
    Actor for training HIPTrack with classification branch
    """
    
    def __init__(self, net, objective, loss_weight, settings, cfg=None, multiFrame=False):
        super().__init__(net, objective, loss_weight, settings, cfg, multiFrame)
        
        # Classification specific settings
        self.cls_weight = loss_weight.get('cls', 1.0)
        self.freeze_cls_epoch = cfg.TRAIN.get('FREEZE_CLS_EPOCH', -1)
    
    def __call__(self, data):
        """
        Forward pass with classification
        
        data should contain:
            - template_images, search_images, template_anno, search_anno (tracking)
            - cls_labels: [num_search_frames, batch] classification labels
        """
        # Check if we should freeze classification branch
        current_epoch = data.get('epoch', 0)
        if self.freeze_cls_epoch > 0 and current_epoch >= self.freeze_cls_epoch:
            # Freeze classification head
            if hasattr(self.net, 'module'):
                for param in self.net.module.cls_head.parameters():
                    param.requires_grad = False
            else:
                for param in self.net.cls_head.parameters():
                    param.requires_grad = False
        
        # Forward pass
        out_dict = self.forward_pass(data)
        
        # Compute losses
        if isinstance(out_dict, list):
            losses = None
            statuses = {
                "Loss/total": None,
                "Loss/giou": None,
                "Loss/l1": None,
                "Loss/location": None,
                "Loss/classification": None,
                "IoU": None,
                "Accuracy": None
            }
            
            for idx, out in enumerate(out_dict):
                partData = {"search_anno": data['search_anno'][idx].unsqueeze(0)}
                
                # Tracking loss
                loss, status = self.compute_losses(out, partData)
                
                # Classification loss
                if 'cls_logits' in out and 'cls_labels' in data:
                    cls_loss, cls_acc = self.compute_classification_loss(
                        out['cls_logits'], 
                        data['cls_labels'][idx]
                    )
                    loss = loss + self.cls_weight * cls_loss
                    status['Loss/classification'] = cls_loss.item()
                    status['Accuracy'] = cls_acc
                else:
                    status['Loss/classification'] = 0.0
                    status['Accuracy'] = 0.0
                
                # Accumulate
                if losses is None:
                    losses = loss
                else:
                    losses += loss
                
                for key, val in status.items():
                    if statuses[key] is None:
                        statuses[key] = val
                    else:
                        statuses[key] += val
            
            # Average IoU and Accuracy
            statuses['IoU'] = statuses['IoU'] / len(out_dict)
            if statuses['Accuracy'] is not None:
                statuses['Accuracy'] = statuses['Accuracy'] / len(out_dict)

            return losses, statuses
        
        else:
            # Single output
            loss, status = self.compute_losses(out_dict, data)
            
            # Add classification loss
            if 'cls_logits' in out_dict and 'cls_labels' in data:
                cls_loss, cls_acc = self.compute_classification_loss(
                    out_dict['cls_logits'], 
                    data['cls_labels']
                )
                loss = loss + self.cls_weight * cls_loss
                status['Loss/classification'] = cls_loss.item()
                status['Accuracy'] = cls_acc
            else:
                status['Loss/classification'] = 0.0
                status['Accuracy'] = 0.0
            
            return loss, status
    
    def compute_classification_loss(self, logits, labels):
        """
        Compute classification loss and accuracy
        
        Args:
            logits: [B, num_classes] - classification logits
            labels: [B] - ground truth class labels
        Returns:
            loss: scalar tensor - classification loss
            accuracy: float - classification accuracy
        """
        # Convert labels to tensor if needed
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long, device=logits.device)
        else:
            labels = labels.to(logits.device)
        
        # Filter out ignore index (-100) for missing annotations
        valid_mask = labels != -100
        
        if valid_mask.sum() == 0:
            # No valid labels, return zero loss
            return torch.tensor(0.0, device=logits.device, requires_grad=True), 0.0
        
        # Compute loss only on valid labels
        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]
        
        cls_loss = self.objective['cls'](valid_logits, valid_labels)
        
        # Compute accuracy
        with torch.no_grad():
            pred_classes = valid_logits.argmax(dim=1)
            accuracy = (pred_classes == valid_labels).float().mean().item()
        
        return cls_loss, accuracy
    
    def forward_pass(self, data):
        """
        Forward pass - same as parent but passes cls_labels
        """
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 5
        
        # Prepare template
        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(
                -1, *data['template_images'].shape[2:]
            )
            template_list.append(template_img_i)
        
        # Prepare search
        search_imgs = [
            data['search_images'][i].view(-1, *data['search_images'].shape[2:])
            for i in range(len(data['search_images']))
        ]
        
        # Box mask and CE keep rate
        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            from lib.utils.ce_utils import generate_mask_cond, adjust_keep_rate
            
            box_mask_z = generate_mask_cond(
                self.cfg, 
                template_list[0].shape[0], 
                template_list[0].device,
                data['template_anno'][0]
            )
            
            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(
                data['epoch'], 
                warmup_epochs=ce_start_epoch,
                total_epochs=ce_start_epoch + ce_warm_epoch,
                ITERS_PER_EPOCH=1,
                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0]
            )
        
        if len(template_list) == 1:
            template_list = template_list[0]
        
        # Forward
        template_annos = data['template_anno']
        
        # Prepare classification labels if available
        cls_labels = data.get('cls_labels', None)
        
        out_dict = self.net(
            template=template_list,
            search=search_imgs,
            ce_template_mask=box_mask_z,
            ce_keep_rate=ce_keep_rate,
            return_last_attn=False,
            template_boxes=template_annos,
            gtBoxes=data['search_anno'],
            # cls_labels=cls_labels
        )
        
        return out_dict

