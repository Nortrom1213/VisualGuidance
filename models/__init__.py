"""
Models package for Visual Guidance System

This package contains the core neural network models:
- STP Detector: For detecting spatial transition points
- MSTP Selector: For selecting the most important STP
- Adapters: For efficient fine-tuning
"""

from .stp_detector import STPDetector, get_stp_detector
from .mstp_selector import MSTPSelectorNet, create_mstp_selector
from .adapters import Adapter, CombinedPredictor, AdapterPredictor

__all__ = [
    "STPDetector",
    "get_stp_detector",
    "MSTPSelectorNet", 
    "create_mstp_selector",
    "Adapter",
    "CombinedPredictor",
    "AdapterPredictor"
]
