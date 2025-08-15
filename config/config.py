"""
Configuration file for Visual Guidance System

This module contains all the configuration parameters for the STP detector,
MSTP selector, and dataset processing.
"""

class Config:
    """Main configuration class for the Visual Guidance System."""
    
    # Model 1: STP Detector Configuration
    STP_DETECTOR_CONFIG = {
        "num_classes": 2,                    # Background + STP
        "adapter_dim": 256,                  # Bottleneck dimension for adapter
        "iou_threshold": 0.5,                # IoU threshold for evaluation
        "score_threshold": 0.5,              # Score threshold for inference
        "learning_rate": 0.005,              # Learning rate for training
        "momentum": 0.9,                     # Momentum for optimizer
        "weight_decay": 0.0005,              # Weight decay for regularization
        "train_val_split": 0.8,              # Training/validation split ratio
    }
    
    # Model 2: MSTP Selector Configuration
    MSTP_SELECTOR_CONFIG = {
        "bottleneck_dim": 256,               # Bottleneck dimension for adapter
        "crop_size": (224, 224),             # Size for candidate region crops
        "global_size": (64, 64),             # Size for global context image
        "feature_dim": 512,                  # Feature dimension from ResNet18
        "hidden_dim": 256,                   # Hidden layer dimension
        "learning_rate": 0.001,              # Learning rate for training
        "batch_size": 8,                     # Batch size for training
    }
    
    # Dataset Configuration
    DATASET_CONFIG = {
        "train_val_split": 0.8,              # Training/validation split ratio
        "batch_size": 8,                     # Batch size for training
        "num_workers": 4,                    # Number of data loading workers
        "pin_memory": True,                  # Pin memory for faster GPU transfer
    }
    
    # Feature Bank Configuration
    FEATURE_BANK_CONFIG = {
        "feature_dim": 512,                  # Feature vector dimension
        "quality_gamma": 0.5,                # Quality score parameter
        "top_k": 1000,                       # Number of top features to keep
    }
    
    # Inference Configuration
    INFERENCE_CONFIG = {
        "use_retrieval": True,               # Whether to use retrieval augmentation
        "retrieval_alpha": 0.7,              # Weight for retrieval vs model scores
        "score_threshold": 0.7,              # Detection confidence threshold
    }
    
    # Supported Game Datasets
    SUPPORTED_GAMES = [
        "DarkSouls1",
        "DarkSouls2", 
        "DarkSouls3",
        "EldenRing",
        "BMW"
    ]
    
    # File Paths
    PATHS = {
        "models": "models/",
        "data": "data/",
        "outputs": "outputs/",
        "logs": "logs/"
    }
