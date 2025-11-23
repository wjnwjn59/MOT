import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """
    Classification head for auxiliary task
    Input: Features from backbone [B, C, H, W] hoặc [B, HW, C]
    Output: Class logits [B, num_classes]
    """
    def __init__(self, in_dim=768, hidden_dim=512, num_classes=10, pooling='avg'):
        """
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of classification classes
            pooling: Pooling method ('avg', 'max', 'attention')
        """
        super().__init__()
        self.pooling_type = pooling
        
        # Pooling layer
        if pooling == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pooling == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif pooling == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(in_dim, in_dim // 4),
                nn.ReLU(),
                nn.Linear(in_dim // 4, 1)
            )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Feature tensor
               - Shape: [B, C, H, W] cho CNN features
               - Shape: [B, HW, C] cho ViT features
        Returns:
            logits: [B, num_classes]
        """
        # Handle different input formats
        if len(x.shape) == 3:  # [B, HW, C]
            B, HW, C = x.shape
            H = W = int(HW ** 0.5)
            x = x.permute(0, 2, 1).reshape(B, C, H, W)  # [B, C, H, W]
        
        # Pooling
        if self.pooling_type == 'attention':
            B, C, H, W = x.shape
            x_flat = x.flatten(2).transpose(1, 2)  # [B, HW, C]
            attn_weights = F.softmax(self.attention(x_flat), dim=1)  # [B, HW, 1]
            x = torch.sum(x_flat * attn_weights, dim=1)  # [B, C]
        else:
            x = self.pool(x)  # [B, C, 1, 1]
            x = x.flatten(1)  # [B, C]
        
        # Classification
        logits = self.classifier(x)  # [B, num_classes]
        
        return logits


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
    in_dim = cfg.MODEL.HIDDEN_DIM
    hidden_dim = cfg.MODEL.CLS_HEAD.HIDDEN_DIM
    pooling = cfg.MODEL.CLS_HEAD.POOLING
    
    return ClassificationHead(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        pooling=pooling
    )

