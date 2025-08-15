"""
Main Spatial Transition Point (MSTP) Selector Model

This module implements a neural network for selecting the most important
spatial transition point from a set of candidate STPs. The model uses
a dual-branch architecture that combines local candidate features with
global context information.

The architecture consists of:
1. Candidate branch: ResNet18 backbone for extracting candidate region features
2. Global branch: Lightweight CNN for global context features
3. Feature fusion: Concatenation followed by adapter module
4. Classifier: MLP for final MSTP selection
"""

import os
import json
import random
import logging
from typing import Dict, List, Tuple, Optional
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as T
import torchvision.models as models

from .adapters import Adapter
from ..config.config import Config


class MSTPSelectorDataset(torch.utils.data.Dataset):
    """
    Dataset class for MSTP selector training and validation.
    
    This dataset loads images and extracts candidate region crops along with
    global context images. It supports random shuffling of candidates to
    improve model robustness.
    
    Args:
        annotations_file (str): Path to JSON file containing candidate annotations
        img_dir (str): Directory containing the image files
        crop_transform (callable): Transform for candidate region crops
        global_transform (callable): Transform for global context images
    """
    
    def __init__(self, annotations_file: str, img_dir: str,
                 crop_transform: Optional[callable] = None,
                 global_transform: Optional[callable] = None):
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.records = json.load(f)
        self.img_dir = img_dir
        
        # Default transforms if none provided
        if crop_transform is None:
            self.crop_transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor()
            ])
        else:
            self.crop_transform = crop_transform
            
        if global_transform is None:
            self.global_transform = T.Compose([
                T.Resize((64, 64)),
                T.ToTensor()
            ])
        else:
            self.global_transform = global_transform
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.records)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            Dict: Sample containing candidates, global image, and ground truth
        """
        record = self.records[idx]
        image_path = os.path.join(self.img_dir, record["image_id"])
        image = Image.open(image_path).convert("RGB")
        
        candidate_boxes = record["candidates"]
        if len(candidate_boxes) == 0:
            return {
                "candidates": [],
                "global": None,
                "gt_index": -1,
                "image_id": record["image_id"]
            }
        
        # Extract candidate region crops
        candidates = []
        for box in candidate_boxes:
            crop = image.crop((box[0], box[1], box[2], box[3]))
            crop = self.crop_transform(crop)
            candidates.append(crop)
        
        # Extract global context image
        global_img = self.global_transform(image)
        gt_index = record["gt_index"]
        
        # Randomly shuffle candidates to improve robustness
        indices = list(range(len(candidates)))
        random.shuffle(indices)
        shuffled_candidates = [candidates[i] for i in indices]
        new_gt_index = indices.index(gt_index)
        
        return {
            "candidates": shuffled_candidates,
            "global": global_img,
            "gt_index": new_gt_index,
            "image_id": record["image_id"]
        }


def selector_collate_fn(batch: List[Dict]) -> List[Dict]:
    """
    Custom collate function for MSTP selector DataLoader.
    
    This function handles variable-length candidate lists in the batch.
    
    Args:
        batch (List[Dict]): List of sample dictionaries
        
    Returns:
        List[Dict]: Batch of samples
    """
    return batch


class MSTPSelectorNet(nn.Module):
    """
    MSTP Selector Network with dual-branch architecture.
    
    This network combines local candidate features with global context
    information to select the most important spatial transition point.
    
    Architecture:
    1. Candidate branch: ResNet18 backbone for candidate region features
    2. Global branch: Lightweight CNN for global context features
    3. Feature fusion: Concatenation followed by adapter module
    4. Classifier: MLP for final MSTP selection
    
    Args:
        bottleneck_dim (int): Bottleneck dimension for adapter
        use_adapter (bool): Whether to use adapter or identity mapping
    """
    
    def __init__(self, bottleneck_dim: int = 256, use_adapter: bool = True):
        super().__init__()
        
        # Candidate branch: ResNet18 without final FC layer
        resnet = models.resnet18(pretrained=True)
        self.candidate_branch = nn.Sequential(*list(resnet.children())[:-1])
        
        # Global branch: Lightweight CNN for global context
        self.global_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 512),
            nn.ReLU(inplace=True)
        )
        
        # Adapter module for feature adaptation
        if use_adapter:
            self.adapter = Adapter(1024, bottleneck_dim)
        else:
            # Identity mapping for ablation studies
            self.adapter = nn.Identity()
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
    
    def forward(self, candidate_imgs: torch.Tensor, 
                global_img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MSTP selector.
        
        Args:
            candidate_imgs (torch.Tensor): Candidate region crops (N, 3, 224, 224)
            global_img (torch.Tensor): Global context image (3, 64, 64)
            
        Returns:
            torch.Tensor: Selection scores for each candidate (N,)
        """
        # Extract candidate features
        candidate_feats = self.candidate_branch(candidate_imgs)  # (N, 512, 1, 1)
        candidate_feats = candidate_feats.view(candidate_feats.size(0), -1)  # (N, 512)
        
        # Extract global features
        global_feats = self.global_branch(global_img.unsqueeze(0))  # (1, 512)
        global_feats = global_feats.expand(candidate_feats.size(0), -1)  # (N, 512)
        
        # Feature fusion
        fused_feats = torch.cat([candidate_feats, global_feats], dim=1)  # (N, 1024)
        
        # Apply adapter
        adapted_feats = self.adapter(fused_feats)  # (N, 1024)
        
        # Final classification
        scores = self.classifier(adapted_feats).squeeze(1)  # (N,)
        
        return scores


def get_selector_transforms() -> Tuple[T.Compose, T.Compose]:
    """
    Get the standard transforms for MSTP selector input.
    
    Returns:
        Tuple[T.Compose, T.Compose]: (crop_transform, global_transform)
    """
    crop_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    global_transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor()
    ])
    
    return crop_transform, global_transform


def train_one_epoch(model: MSTPSelectorNet, data_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, loss_fn: nn.Module) -> float:
    """
    Train the MSTP selector for one epoch.
    
    Args:
        model (MSTPSelectorNet): Model to train
        data_loader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to train on
        epoch (int): Current epoch number
        loss_fn (nn.Module): Loss function
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    num_samples = 0
    
    for sample in data_loader:
        for s in sample:
            candidates = s["candidates"]
            global_img = s["global"]
            gt_index = s["gt_index"]
            
            # Skip invalid samples
            if len(candidates) == 0 or global_img is None:
                continue
            
            # Prepare inputs
            x = torch.stack(candidates).to(device)
            global_tensor = global_img.to(device)
            
            # Forward pass
            logits = model(x, global_tensor)
            logits = logits.unsqueeze(0)  # Add batch dimension
            
            # Prepare target
            target = torch.tensor([gt_index], dtype=torch.long, device=device)
            
            # Compute loss
            loss = loss_fn(logits, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_samples += 1
    
    avg_loss = running_loss / max(num_samples, 1)
    print(f"Epoch {epoch} Training Loss: {avg_loss:.4f}")
    
    return avg_loss


def evaluate_selector(model: MSTPSelectorNet, data_loader: torch.utils.data.DataLoader,
                     device: torch.device) -> Dict[str, float]:
    """
    Evaluate MSTP selector performance.
    
    Args:
        model (MSTPSelectorNet): Trained model to evaluate
        data_loader (DataLoader): Validation data loader
        device (torch.device): Device to run evaluation on
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    model.eval()
    total_samples = 0
    correct_predictions = 0
    
    with torch.no_grad():
        for sample in data_loader:
            for s in sample:
                candidates = s["candidates"]
                global_img = s["global"]
                gt_index = s["gt_index"]
                
                # Skip invalid samples
                if len(candidates) == 0 or global_img is None:
                    continue
                
                # Prepare inputs
                x = torch.stack(candidates).to(device)
                global_tensor = global_img.to(device)
                
                # Forward pass
                logits = model(x, global_tensor)
                pred_index = torch.argmax(logits).item()
                
                total_samples += 1
                if pred_index == gt_index:
                    correct_predictions += 1
    
    # Calculate accuracy
    accuracy = correct_predictions / max(total_samples, 1)
    
    return {
        "accuracy": accuracy,
        "total_samples": total_samples,
        "correct_predictions": correct_predictions
    }


def create_mstp_selector(bottleneck_dim: int = 256, 
                        use_adapter: bool = True) -> MSTPSelectorNet:
    """
    Factory function to create an MSTP selector instance.
    
    Args:
        bottleneck_dim (int): Bottleneck dimension for adapter
        use_adapter (bool): Whether to use adapter modules
        
    Returns:
        MSTPSelectorNet: Configured MSTP selector model
    """
    return MSTPSelectorNet(bottleneck_dim, use_adapter)
