"""
Spatial Transition Point (STP) Detector Model

This module implements a two-stage object detection model for identifying
spatial transition points in game screenshots. The model is based on
Faster R-CNN with Feature Pyramid Network (FPN) and includes adapter
modules for efficient fine-tuning.

The model detects all STPs (both MSTP and regular STP) as a single class,
enabling the downstream MSTP selector to choose the most important one.
"""

import os
import cv2
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
import torchvision.transforms as T

from .adapters import CombinedPredictor, AdapterPredictor
from config.config import Config


class STPDataset(torch.utils.data.Dataset):
    """
    Dataset class for STP detection training and validation.
    
    This dataset loads images and their corresponding STP annotations,
    preparing them for training the Faster R-CNN model.
    
    Args:
        annotations_file (str): Path to JSON file containing annotations
        img_dir (str): Directory containing the image files
        transform (callable): Optional transform to be applied to images
    """
    
    def __init__(self, annotations_file: str, img_dir: str, 
                 transform: Optional[callable] = None):
        # Handle both file path and loaded data
        if isinstance(annotations_file, str):
            with open(annotations_file, 'r', encoding='utf-8') as f:
                self.records = json.load(f)
        else:
            # Assume annotations_file is already loaded data
            self.records = annotations_file
        self.img_dir = img_dir
        
        # Set default transform if none provided
        if transform is None:
            self.transform = T.Compose([T.ToTensor()])
        else:
            self.transform = transform
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.records)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image_tensor, target_dict)
        """
        record = self.records[idx]
        image_path = os.path.join(self.img_dir, record["image_id"])
        image = Image.open(image_path).convert("RGB")
        
        # Convert PIL image to tensor (always apply transform)
        image = self.transform(image)
        
        # Combine MSTP and STP into boxes
        boxes = []
        if "MSTP" in record and record["MSTP"]:
            boxes.append(record["MSTP"])
        if "STP" in record and record["STP"]:
            boxes.extend(record["STP"])
        
        # Calculate areas for bounding boxes
        areas = []
        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            areas.append(area)
        
        # Prepare target dictionary
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.ones(len(boxes), dtype=torch.long) if boxes else torch.empty((0,), dtype=torch.long),
            "image_id": torch.tensor([idx]),
            "area": torch.tensor(areas, dtype=torch.float32) if areas else torch.empty((0,), dtype=torch.float32),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.long) if boxes else torch.empty((0,), dtype=torch.long)
        }
        
        return image, target


def collate_fn(batch: List[Tuple]) -> Tuple[List[torch.Tensor], List[Dict]]:
    """
    Custom collate function for DataLoader.
    
    This function handles variable-length bounding box lists in the batch.
    
    Args:
        batch (List[Tuple]): List of (image, target) tuples
        
    Returns:
        tuple: (images_list, targets_list)
    """
    return tuple(zip(*batch))


class STPDetector(nn.Module):
    """
    STP Detector based on Faster R-CNN with FPN.
    
    This model extends the standard Faster R-CNN architecture with
    adapter modules in the prediction head for efficient fine-tuning.
    
    Args:
        num_classes (int): Number of classes (background + STP)
        adapter_dim (int): Bottleneck dimension for adapter modules
        use_adapter (bool): Whether to use adapter or standard predictor
    """
    
    def __init__(self, num_classes: int = 2, adapter_dim: int = 256, 
                 use_adapter: bool = True):
        super().__init__()
        
        # Load pre-trained Faster R-CNN with FPN
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Get the original predictor's input feature dimension
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        
        if use_adapter:
            # Create new predictor with adapter
            new_predictor = FastRCNNPredictor(in_features, num_classes)
            self.model.roi_heads.box_predictor = CombinedPredictor(
                new_predictor, in_features, adapter_dim, use_adapter=True
            )
        else:
            # Use standard predictor
            new_predictor = FastRCNNPredictor(in_features, num_classes)
            self.model.roi_heads.box_predictor = new_predictor
    
    def forward(self, images: List[torch.Tensor], 
                targets: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Forward pass of the STP detector.
        
        Args:
            images (List[torch.Tensor]): List of input images
            targets (Optional[List[Dict]]): Ground truth targets for training
            
        Returns:
            List[Dict]: List of detection results
        """
        return self.model(images, targets)


def get_stp_detector(num_classes: int = 2, adapter_dim: int = 256, 
                     use_adapter: bool = True) -> STPDetector:
    """
    Factory function to create an STP detector instance.
    
    Args:
        num_classes (int): Number of classes
        adapter_dim (int): Bottleneck dimension for adapter
        use_adapter (bool): Whether to use adapter modules
        
    Returns:
        STPDetector: Configured STP detector model
    """
    return STPDetector(num_classes, adapter_dim, use_adapter)


def get_detector_transform() -> T.Compose:
    """
    Get the standard transform for STP detector input.
    
    Returns:
        T.Compose: Transform pipeline for input images
    """
    return T.Compose([T.ToTensor()])


def run_stp_detector(pil_img: Image.Image, model: STPDetector, 
                     device: torch.device, score_threshold: float = 0.5) -> np.ndarray:
    """
    Run STP detection on a single image.
    
    Args:
        pil_img (Image.Image): Input PIL image
        model (STPDetector): Trained STP detector model
        device (torch.device): Device to run inference on
        score_threshold (float): Confidence threshold for detections
        
    Returns:
        np.ndarray: Array of detected bounding boxes
    """
    # Prepare input
    img_tensor = get_detector_transform()(pil_img).to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        predictions = model([img_tensor])[0]
    
    # Filter by confidence threshold
    boxes = predictions["boxes"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()
    keep_indices = np.where(scores >= score_threshold)[0]
    
    return boxes[keep_indices]


def evaluate_stp_detector(model: STPDetector, data_loader: torch.utils.data.DataLoader,
                         device: torch.device, iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate STP detector performance.
    
    Args:
        model (STPDetector): Trained model to evaluate
        data_loader (DataLoader): Validation data loader
        device (torch.device): Device to run evaluation on
        iou_threshold (float): IoU threshold for correct detection
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    model.eval()
    total_predictions = 0
    total_ground_truth = 0
    correct_detections = 0
    total_iou = 0.0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            predictions = model(images)
            
            for pred, target in zip(predictions, targets):
                pred_boxes = pred["boxes"].cpu().numpy()
                target_boxes = target["boxes"].cpu().numpy()
                
                total_predictions += len(pred_boxes)
                total_ground_truth += len(target_boxes)
                
                # Calculate IoU for each prediction-target pair
                for pred_box in pred_boxes:
                    max_iou = 0.0
                    for target_box in target_boxes:
                        iou = calculate_iou(pred_box, target_box)
                        max_iou = max(max_iou, iou)
                    
                    if max_iou >= iou_threshold:
                        correct_detections += 1
                        total_iou += max_iou
    
    # Calculate metrics
    precision = correct_detections / max(total_predictions, 1)
    recall = correct_detections / max(total_ground_truth, 1)
    mean_iou = total_iou / max(correct_detections, 1)
    
    return {
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "f1_score": 2 * (precision * recall) / max(precision + recall, 1e-8)
    }


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1 (np.ndarray): First bounding box [x1, y1, x2, y2]
        box2 (np.ndarray): Second bounding box [x1, y1, x2, y2]
        
    Returns:
        float: IoU value between 0 and 1
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate areas
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / max(union, 1e-8)
