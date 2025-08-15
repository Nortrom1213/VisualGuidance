"""
Feature Bank for Retrieval-Augmented Generation (RAG)

This module implements a feature bank system that stores and retrieves
visual features for candidate regions. It enables retrieval augmentation
during MSTP selection by computing similarity scores between query
features and stored features.
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T


class FeatureBank:
    """
    Feature bank for storing and retrieving visual features.
    
    This class maintains a database of visual features extracted from
    candidate regions, enabling similarity-based retrieval during
    inference.
    
    Args:
        feature_dim (int): Dimension of feature vectors
        quality_gamma (float): Parameter for quality score computation
        top_k (int): Number of top features to keep
    """
    
    def __init__(self, feature_dim: int = 512, quality_gamma: float = 0.5, 
                 top_k: int = 1000):
        self.feature_dim = feature_dim
        self.quality_gamma = quality_gamma
        self.top_k = top_k
        self.features = []
        self.metadata = []
    
    def add_feature(self, feature: np.ndarray, metadata: Dict) -> None:
        """
        Add a feature vector to the bank.
        
        Args:
            feature (np.ndarray): Feature vector to add
            metadata (Dict): Associated metadata (image_id, bbox, etc.)
        """
        if len(feature) != self.feature_dim:
            raise ValueError(f"Feature dimension mismatch: expected {self.feature_dim}, got {len(feature)}")
        
        # Compute quality score
        quality_score = self._compute_quality_score(feature)
        
        # Store feature and metadata
        self.features.append(feature)
        self.metadata.append({
            **metadata,
            "quality_score": quality_score
        })
    
    def _compute_quality_score(self, feature: np.ndarray) -> float:
        """
        Compute quality score for a feature vector.
        
        The quality score is based on feature magnitude and variance,
        indicating how informative the feature is.
        
        Args:
            feature (np.ndarray): Feature vector
            
        Returns:
            float: Quality score
        """
        # L2 norm of the feature
        magnitude = np.linalg.norm(feature)
        
        # Standard deviation of the feature
        std_dev = np.std(feature)
        
        # Combined quality score
        quality_score = magnitude + self.quality_gamma * std_dev
        
        return quality_score
    
    def build_from_annotations(self, annotations_file: str, images_dir: str,
                              feature_extractor: Optional[nn.Module] = None) -> None:
        """
        Build feature bank from annotation file and images.
        
        Args:
            annotations_file (str): Path to JSON annotations file
            images_dir (str): Directory containing images
            feature_extractor (Optional[nn.Module]): Pre-trained feature extractor
        """
        if feature_extractor is None:
            feature_extractor = self._get_default_feature_extractor()
        
        # Load annotations
        with open(annotations_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        print(f"Building feature bank from {len(annotations)} annotations...")
        
        for i, annotation in enumerate(annotations):
            if i % 100 == 0:
                print(f"Processing annotation {i}/{len(annotations)}")
            
            image_id = annotation["image_id"]
            image_path = os.path.join(images_dir, image_id)
            
            if not os.path.exists(image_path):
                continue
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Process MSTP annotations
            if "MSTP" in annotation:
                mstp_boxes = annotation["MSTP"]
                if isinstance(mstp_boxes[0], list):
                    # Multiple MSTP boxes
                    for box in mstp_boxes:
                        self._process_region(image, box, image_id, "MSTP", feature_extractor)
                else:
                    # Single MSTP box
                    self._process_region(image, mstp_boxes, image_id, "MSTP", feature_extractor)
            
            # Process STP annotations
            if "STP" in annotation:
                stp_boxes = annotation["STP"]
                for box in stp_boxes:
                    self._process_region(image, box, image_id, "STP", feature_extractor)
        
        # Sort by quality score and keep top-k
        self._sort_and_filter()
        print(f"Feature bank built with {len(self.features)} features")
    
    def _process_region(self, image: Image.Image, box: List[int], image_id: str,
                       region_type: str, feature_extractor: nn.Module) -> None:
        """
        Process a single region and add its features to the bank.
        
        Args:
            image (Image.Image): Source image
            box (List[int]): Bounding box [x1, y1, x2, y2]
            image_id (str): Image identifier
            region_type (str): Type of region (MSTP or STP)
            feature_extractor (nn.Module): Feature extraction model
        """
        try:
            # Extract region crop
            crop = image.crop((box[0], box[1], box[2], box[3]))
            
            # Extract features
            features = self._extract_features(crop, feature_extractor)
            
            # Add to bank
            metadata = {
                "image_id": image_id,
                "bbox": box,
                "region_type": region_type
            }
            
            self.add_feature(features, metadata)
            
        except Exception as e:
            print(f"Error processing region {box} from {image_id}: {e}")
    
    def _extract_features(self, image_crop: Image.Image, 
                         feature_extractor: nn.Module) -> np.ndarray:
        """
        Extract features from an image crop.
        
        Args:
            image_crop (Image.Image): Image crop to process
            feature_extractor (nn.Module): Feature extraction model
            
        Returns:
            np.ndarray: Extracted feature vector
        """
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])
        
        input_tensor = transform(image_crop).unsqueeze(0)
        
        with torch.no_grad():
            features = feature_extractor(input_tensor)
            features = features.view(-1).cpu().numpy()
        
        return features
    
    def _get_default_feature_extractor(self) -> nn.Module:
        """
        Get default pre-trained ResNet18 feature extractor.
        
        Returns:
            nn.Module: Pre-trained feature extractor
        """
        model = torchvision.models.resnet18(pretrained=True)
        # Remove final FC layer
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        feature_extractor.eval()
        return feature_extractor
    
    def _sort_and_filter(self) -> None:
        """Sort features by quality score and keep top-k."""
        if len(self.features) <= self.top_k:
            return
        
        # Sort by quality score (descending)
        sorted_indices = np.argsort([meta["quality_score"] for meta in self.metadata])[::-1]
        
        # Keep top-k features
        self.features = [self.features[i] for i in sorted_indices[:self.top_k]]
        self.metadata = [self.metadata[i] for i in sorted_indices[:self.top_k]]
    
    def compute_similarity_score(self, query_feature: np.ndarray) -> float:
        """
        Compute similarity score between query and stored features.
        
        Args:
            query_feature (np.ndarray): Query feature vector
            
        Returns:
            float: Maximum similarity score
        """
        if len(self.features) == 0:
            return 0.0
        
        # Compute cosine similarity with all stored features
        similarities = []
        for stored_feature in self.features:
            similarity = self._cosine_similarity(query_feature, stored_feature)
            similarities.append(similarity)
        
        # Return maximum similarity
        return max(similarities) if similarities else 0.0
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two feature vectors.
        
        Args:
            a (np.ndarray): First feature vector
            b (np.ndarray): Second feature vector
            
        Returns:
            float: Cosine similarity score
        """
        # Normalize vectors
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        
        if a_norm == 0 or b_norm == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(a, b) / (a_norm * b_norm)
        
        # Clamp to [-1, 1] to handle numerical errors
        similarity = np.clip(similarity, -1.0, 1.0)
        
        return similarity
    
    def save(self, filepath: str) -> None:
        """
        Save feature bank to file.
        
        Args:
            filepath (str): Path to save the feature bank
        """
        data = {
            "feature_dim": self.feature_dim,
            "quality_gamma": self.quality_gamma,
            "top_k": self.top_k,
            "features": self.features,
            "metadata": self.metadata
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Feature bank saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureBank':
        """
        Load feature bank from file.
        
        Args:
            filepath (str): Path to load the feature bank from
            
        Returns:
            FeatureBank: Loaded feature bank instance
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Feature bank file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create instance
        feature_bank = cls(
            feature_dim=data["feature_dim"],
            quality_gamma=data["quality_gamma"],
            top_k=data["top_k"]
        )
        
        # Restore data
        feature_bank.features = data["features"]
        feature_bank.metadata = data["metadata"]
        
        print(f"Feature bank loaded from {filepath} with {len(feature_bank.features)} features")
        return feature_bank
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the feature bank.
        
        Returns:
            Dict: Statistics including feature count, quality scores, etc.
        """
        if len(self.features) == 0:
            return {"total_features": 0}
        
        quality_scores = [meta["quality_score"] for meta in self.metadata]
        region_types = [meta["region_type"] for meta in self.metadata]
        
        stats = {
            "total_features": len(self.features),
            "feature_dimension": self.feature_dim,
            "quality_score_mean": np.mean(quality_scores),
            "quality_score_std": np.std(quality_scores),
            "quality_score_min": np.min(quality_scores),
            "quality_score_max": np.max(quality_scores),
            "mstp_count": region_types.count("MSTP"),
            "stp_count": region_types.count("STP")
        }
        
        return stats


def build_feature_bank_cli():
    """Command-line interface for building feature banks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Feature Bank for Visual Guidance")
    
    parser.add_argument("--annotations", type=str, required=True,
                       help="Path to annotations JSON file")
    parser.add_argument("--images_dir", type=str, required=True,
                       help="Directory containing images")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for feature bank")
    parser.add_argument("--feature_dim", type=int, default=512,
                       help="Feature vector dimension")
    parser.add_argument("--quality_gamma", type=float, default=0.5,
                       help="Quality score parameter")
    parser.add_argument("--top_k", type=int, default=1000,
                       help="Number of top features to keep")
    
    args = parser.parse_args()
    
    # Create feature bank
    feature_bank = FeatureBank(
        feature_dim=args.feature_dim,
        quality_gamma=args.quality_gamma,
        top_k=args.top_k
    )
    
    # Build from annotations
    feature_bank.build_from_annotations(args.annotations, args.images_dir)
    
    # Save feature bank
    feature_bank.save(args.output)
    
    # Print statistics
    stats = feature_bank.get_statistics()
    print("\nFeature Bank Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    build_feature_bank_cli()
