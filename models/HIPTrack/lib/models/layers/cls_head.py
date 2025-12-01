import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """
    Classification head for multi-task learning
    Uses global pooled features from backbone (same features as bbox head)
    Input: Pooled features [B, C]
    Output: Class logits [B, num_classes]
    """
    def __init__(self, in_dim=768, hidden_dim=512, num_classes=10, dropout=0.1):
        """
        Args:
            in_dim: Input feature dimension (from backbone)
            hidden_dim: Hidden layer dimension for classification
            num_classes: Number of classification classes
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        # Simple MLP: input -> hidden -> classes
        self.cls_projection = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for all linear layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """
        Forward pass for classification
        
        Args:
            features: Pooled features from backbone [B, C]
        
        Returns:
            cls_logits: Classification logits [B, num_classes]
        """
        # Project to hidden dimension
        x = self.cls_projection(features)  # [B, hidden_dim]
        x = F.relu(x)
        x = self.dropout(x)
        
        # Classification prediction
        cls_logits = self.classifier(x)  # [B, num_classes]
        
        return cls_logits


class MultiScaleClassificationHead(nn.Module):
    """
    Multi-scale classification head that combines features from different scales
    """
    def __init__(self, in_dims=[384, 768], hidden_dim=512, num_classes=10):
        super().__init__()
        self.heads = nn.ModuleList([
            ClassificationHead(in_dim, hidden_dim // len(in_dims), num_classes)
            for in_dim in in_dims
        ])
        
        # Fusion layer
        self.fusion = nn.Linear(num_classes * len(in_dims), num_classes)
    
    def forward(self, features):
        """
        Args:
            features: List of feature tensors from different scales
        Returns:
            logits: [B, num_classes]
        """
        logits_list = []
        for feat, head in zip(features, self.heads):
            logits_list.append(head(feat))
        
        # Concatenate and fuse
        combined = torch.cat(logits_list, dim=1)
        final_logits = self.fusion(combined)
        
        return final_logits


def build_cls_head(cfg):
    """
    Build classification head based on config
    """
    num_classes = cfg.MODEL.CLS_HEAD.NUM_CLASSES
    in_dim = cfg.MODEL.HIDDEN_DIM  # Feature dimension from backbone
    hidden_dim = cfg.MODEL.CLS_HEAD.HIDDEN_DIM
    dropout = cfg.MODEL.CLS_HEAD.DROPOUT
    
    return ClassificationHead(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout
    )

