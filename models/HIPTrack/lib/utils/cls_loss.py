import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """
    Standard cross-entropy loss for classification
    """
    def __init__(self, weight=None, ignore_index=-100):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            weight=weight, 
            ignore_index=ignore_index
        )
    
    def forward(self, logits, labels):
        """
        Args:
            logits: [B, num_classes]
            labels: [B]
        Returns:
            loss: scalar
        """
        return self.loss_fn(logits, labels)


class FocalLossClassification(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, logits, labels):
        """
        Args:
            logits: [B, num_classes]
            labels: [B]
        Returns:
            loss: scalar
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(
            logits, labels, 
            reduction='none', 
            ignore_index=self.ignore_index
        )
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Handle ignore index
        mask = labels != self.ignore_index
        focal_loss = focal_loss[mask].mean() if mask.sum() > 0 else focal_loss.mean()
        
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing for better generalization
    """
    def __init__(self, num_classes, smoothing=0.1, ignore_index=-100):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits, labels):
        """
        Args:
            logits: [B, num_classes]
            labels: [B]
        Returns:
            loss: scalar
        """
        with torch.no_grad():
            # Create smoothed labels
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, labels.unsqueeze(1), self.confidence)
            
            # Handle ignore index
            mask = labels == self.ignore_index
            true_dist[mask] = 0
        
        # Compute KL divergence
        log_probs = F.log_softmax(logits, dim=1)
        loss = -torch.sum(true_dist * log_probs, dim=1)
        
        # Apply mask
        mask = labels != self.ignore_index
        loss = loss[mask].mean() if mask.sum() > 0 else loss.mean()
        
        return loss


def build_cls_loss(cfg):
    """
    Build classification loss based on config
    """
    loss_type = cfg.TRAIN.CLS_LOSS_TYPE
    num_classes = cfg.MODEL.CLS_HEAD.NUM_CLASSES
    
    if loss_type == "CE":
        return ClassificationLoss()
    elif loss_type == "FOCAL":
        alpha = cfg.TRAIN.get('CLS_FOCAL_ALPHA', 0.25)
        gamma = cfg.TRAIN.get('CLS_FOCAL_GAMMA', 2.0)
        return FocalLossClassification(alpha=alpha, gamma=gamma)
    elif loss_type == "LABEL_SMOOTH":
        smoothing = cfg.TRAIN.get('CLS_LABEL_SMOOTHING', 0.1)
        return LabelSmoothingLoss(num_classes=num_classes, smoothing=smoothing)
    else:
        raise ValueError(f"Unknown classification loss type: {loss_type}")

