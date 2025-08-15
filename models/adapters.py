"""
Adapter modules for fine-tuning pre-trained models

This module implements various adapter architectures that can be inserted
into pre-trained models to enable efficient fine-tuning while preserving
the original model weights.
"""

import torch
import torch.nn as nn
from typing import Optional


class Adapter(nn.Module):
    """
    Standard bottleneck adapter module.
    
    This adapter implements a bottleneck architecture with down-projection,
    non-linearity, and up-projection layers. It adds the adapted features
    to the original input through a residual connection.
    
    Args:
        in_features (int): Input feature dimension
        bottleneck_dim (int): Bottleneck dimension for the adapter
        activation (nn.Module): Activation function (default: ReLU)
    """
    
    def __init__(self, in_features: int, bottleneck_dim: int = 256, 
                 activation: nn.Module = nn.ReLU()):
        super().__init__()
        self.down = nn.Linear(in_features, bottleneck_dim)
        self.activation = activation
        self.up = nn.Linear(bottleneck_dim, in_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the adapter.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, in_features)
            
        Returns:
            torch.Tensor: Output tensor with residual connection
        """
        return x + self.up(self.activation(self.down(x)))


class IdentityAdapter(nn.Module):
    """
    Identity adapter that passes through input unchanged.
    
    This adapter is useful for ablation studies or when no adaptation
    is needed during training.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - returns input unchanged.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Same as input
        """
        return x


class CombinedPredictor(nn.Module):
    """
    Wrapper that combines a predictor with an adapter.
    
    This class wraps an existing predictor (e.g., FastRCNNPredictor) and
    inserts an adapter before the prediction layers. This enables fine-tuning
    of the adapter while keeping the original predictor weights frozen.
    
    Args:
        predictor (nn.Module): Original predictor module
        in_features (int): Input feature dimension
        adapter_dim (int): Bottleneck dimension for adapter
        use_adapter (bool): Whether to use adapter or identity
    """
    
    def __init__(self, predictor: nn.Module, in_features: int, 
                 adapter_dim: int = 256, use_adapter: bool = True):
        super().__init__()
        self.predictor = predictor
        
        # Choose between adapter and identity
        if use_adapter:
            self.adapter = Adapter(in_features, adapter_dim)
        else:
            self.adapter = IdentityAdapter()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adapter and predictor.
        
        Args:
            x (torch.Tensor): Input features
            
        Returns:
            torch.Tensor: Predictor output
        """
        x = self.adapter(x)
        return self.predictor(x)


class AdapterPredictor(nn.Module):
    """
    Custom predictor with built-in adapter for STP detection.
    
    This predictor implements the classification and regression heads
    for STP detection with an integrated adapter module.
    
    Args:
        in_features (int): Input feature dimension
        num_classes (int): Number of classes (background + STP)
        bottleneck_dim (int): Bottleneck dimension for adapter
    """
    
    def __init__(self, in_features: int, num_classes: int, 
                 bottleneck_dim: int = 256):
        super().__init__()
        self.adapter = Adapter(in_features, bottleneck_dim)
        self.cls_score = nn.Linear(in_features, num_classes)
        self.bbox_pred = nn.Linear(in_features, num_classes * 4)
        
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through adapter and prediction heads.
        
        Args:
            x (torch.Tensor): Input features of shape (N, in_features)
            
        Returns:
            tuple: (classification_scores, bbox_predictions)
        """
        x = self.adapter(x)
        scores = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return scores, bbox_pred
