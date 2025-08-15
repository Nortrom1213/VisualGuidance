"""
Visual Guidance System

A comprehensive system for detecting and selecting spatial transition points
in game screenshots using deep learning with adapter-based fine-tuning.

This package provides:
- STP Detector: Faster R-CNN-based detection of spatial transition points
- MSTP Selector: Neural network for selecting the most important STP
- Inference Pipeline: Complete pipeline for end-to-end processing
- Feature Bank: Retrieval augmentation system for improved performance
- Training Scripts: Complete training workflows for both models

Author: [Your Name]
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "[Your Name]"

from .models.stp_detector import STPDetector, get_stp_detector
from .models.mstp_selector import MSTPSelectorNet, create_mstp_selector
from .pipeline.inference import VisualGuidancePipeline
from .utils.feature_bank import FeatureBank

__all__ = [
    "STPDetector",
    "get_stp_detector", 
    "MSTPSelectorNet",
    "create_mstp_selector",
    "VisualGuidancePipeline",
    "FeatureBank"
]
