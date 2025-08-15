"""
Pipeline package for Visual Guidance System

This package contains the inference and processing pipelines:
- Inference: Complete end-to-end processing pipeline
- Realtime: Real-time inference capabilities
"""

from .inference import VisualGuidancePipeline

__all__ = [
    "VisualGuidancePipeline"
]
