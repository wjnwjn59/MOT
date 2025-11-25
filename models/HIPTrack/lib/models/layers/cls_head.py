import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """
    Classification head for auxiliary task (SimTrackMod architecture)
    Uses CLS token from ViT backbone as global feature representation
    Input: CLS token features [B, C]
    Output: Class logits [B, num_classes] and fusion features [B, bottleneck_dim]
    """
    def __init__(self, in_dim=768, hidden_dim=512, num_classes=10, bottleneck_dim=256):
        """
        Args:
            in_dim: Input CLS token dimension (from ViT backbone)
            hidden_dim: Hidden layer dimension for classification
            num_classes: Number of classification classes
            bottleneck_dim: Dimension for fusion features (to feed back to box head)
        """
        super().__init__()
        
        # Projection layer: CLS token -> hidden dimension
        self.cls_projection = nn.Linear(in_dim, hidden_dim)
        
        # Dropout for regularization (SimTrackMod uses 0.1)
        self.dropout = nn.Dropout(0.1)
        
        # Classifier: hidden -> num_classes
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Fusion layer: project hidden features back to bottleneck dimension
        # These features will be fused with spatial features for box prediction
        self.fusion_layer = nn.Linear(hidden_dim, bottleneck_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for all linear layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, cls_token):
        """
        Forward pass using CLS token from ViT backbone
        
        Args:
            cls_token: CLS token features from ViT [B, C]
        
        Returns:
            dict containing:
                - 'cls_logits': Classification logits [B, num_classes]
                - 'cls_features': Hidden features [B, hidden_dim] 
                - 'fusion_features': Features for box head fusion [B, bottleneck_dim]
        """
        # Project CLS token to hidden dimension
        cls_features = self.cls_projection(cls_token)  # [B, hidden_dim]
        cls_features = F.relu(cls_features)
        cls_features = self.dropout(cls_features)
        
        # Classification prediction
        cls_logits = self.classifier(cls_features)  # [B, num_classes]
        
        # Fusion features for box head
        fusion_features = self.fusion_layer(cls_features)  # [B, bottleneck_dim]
        
        return {
            'cls_logits': cls_logits,
            'cls_features': cls_features,
            'fusion_features': fusion_features
        }


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
    in_dim = cfg.MODEL.HIDDEN_DIM  # CLS token dimension from backbone
    hidden_dim = cfg.MODEL.CLS_HEAD.HIDDEN_DIM
    
    # Bottleneck dimension for fusion (should match backbone bottleneck)
    bottleneck_dim = getattr(cfg.MODEL, 'BOTTLENECK_DIM', 256)
    
    return ClassificationHead(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        bottleneck_dim=bottleneck_dim
    )

