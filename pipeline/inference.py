"""
Inference Pipeline for Visual Guidance System

This module implements the complete inference pipeline that combines
STP detection and MSTP selection with optional retrieval augmentation.
The pipeline processes images to identify the most important spatial
transition point for navigation guidance.
"""

import os
import cv2
import json
import argparse
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.models as models

from ..models.stp_detector import STPDetector, get_stp_detector, get_detector_transform
from ..models.mstp_selector import MSTPSelectorNet, get_selector_transforms
from ..utils.feature_bank import FeatureBank
from ..config.config import Config


class VisualGuidancePipeline:
    """
    Complete pipeline for visual guidance inference.
    
    This class integrates STP detection and MSTP selection with optional
    retrieval augmentation to provide comprehensive spatial transition
    point analysis.
    
    Args:
        detector_model_path (str): Path to trained STP detector checkpoint
        selector_model_path (str): Path to trained MSTP selector checkpoint
        use_retrieval (bool): Whether to use retrieval augmentation
        retrieval_bank_file (str): Path to feature bank for retrieval
        alpha (float): Weight for retrieval vs model scores
        device (torch.device): Device to run inference on
    """
    
    def __init__(self, detector_model_path: str, selector_model_path: str,
                 use_retrieval: bool = True, retrieval_bank_file: str = "feature_bank.pkl",
                 alpha: float = 0.7, device: Optional[torch.device] = None):
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.use_retrieval = use_retrieval
        self.alpha = alpha
        
        # Load models
        self.detector = self._load_detector(detector_model_path)
        self.selector = self._load_selector(selector_model_path)
        
        # Get transforms
        self.detector_transform = get_detector_transform()
        self.crop_transform, self.global_transform = get_selector_transforms()
        
        # Load retrieval bank if needed
        self.feature_bank = None
        if use_retrieval:
            self.feature_bank = FeatureBank.load(retrieval_bank_file)
            
        # Feature extractor for retrieval
        self.feature_extractor = self._get_feature_extractor()
    
    def _load_detector(self, model_path: str) -> STPDetector:
        """Load and configure STP detector."""
        detector = get_stp_detector(num_classes=2, adapter_dim=256, use_adapter=True)
        
        # Load checkpoint with key remapping for adapter compatibility
        checkpoint = torch.load(model_path, map_location=self.device)
        remapped_checkpoint = self._remap_detector_keys(checkpoint)
        
        missing, unexpected = detector.load_state_dict(remapped_checkpoint, strict=False)
        print(f"Loaded STP detector from {model_path}")
        if missing:
            print(f"  Missing keys (adapter init): {missing}")
        if unexpected:
            print(f"  Unexpected keys ignored: {unexpected}")
            
        detector.to(self.device)
        return detector
    
    def _load_selector(self, model_path: str) -> MSTPSelectorNet:
        """Load and configure MSTP selector."""
        selector = MSTPSelectorNet(bottleneck_dim=256, use_adapter=True)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        missing, unexpected = selector.load_state_dict(checkpoint, strict=False)
        
        print(f"Loaded MSTP selector from {model_path}")
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys ignored: {unexpected}")
            
        selector.to(self.device)
        return selector
    
    def _remap_detector_keys(self, checkpoint: Dict) -> Dict:
        """Remap checkpoint keys for adapter compatibility."""
        remapped = {}
        for key, value in checkpoint.items():
            if key.startswith("roi_heads.box_predictor.cls_score"):
                new_key = key.replace("roi_heads.box_predictor.", 
                                    "roi_heads.box_predictor.predictor.")
            elif key.startswith("roi_heads.box_predictor.bbox_pred"):
                new_key = key.replace("roi_heads.box_predictor.", 
                                    "roi_heads.box_predictor.predictor.")
            else:
                new_key = key
            remapped[new_key] = value
        return remapped
    
    def _get_feature_extractor(self) -> nn.Module:
        """Get feature extractor for retrieval augmentation."""
        resnet = models.resnet18(pretrained=True)
        return nn.Sequential(*list(resnet.children())[:-1])
    
    def process_single_image(self, image_path: str, score_threshold: float = 0.7) -> Tuple[Optional[np.ndarray], np.ndarray, int]:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path (str): Path to input image
            score_threshold (float): Confidence threshold for STP detection
            
        Returns:
            Tuple: (processed_image, detected_boxes, selected_mstp_index)
        """
        # Load and prepare image
        pil_image = Image.open(image_path).convert("RGB")
        cv_image = cv2.imread(image_path)
        
        if cv_image is None:
            return None, np.array([]), -1
        
        # Stage 1: STP Detection
        detected_boxes = self._detect_stps(pil_image, score_threshold)
        if len(detected_boxes) == 0:
            return None, np.array([]), -1
        
        # Stage 2: MSTP Selection
        selected_index = self._select_mstp(pil_image, detected_boxes)
        
        # Stage 3: Visualization
        processed_image = self._visualize_results(cv_image, detected_boxes, selected_index)
        
        return processed_image, detected_boxes, selected_index
    
    def _detect_stps(self, pil_image: Image.Image, score_threshold: float) -> np.ndarray:
        """Detect STPs using the STP detector."""
        self.detector.eval()
        
        with torch.no_grad():
            img_tensor = self.detector_transform(pil_image).to(self.device)
            predictions = self.detector([img_tensor])[0]
        
        # Filter by confidence threshold
        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        keep_indices = np.where(scores >= score_threshold)[0]
        
        return boxes[keep_indices]
    
    def _select_mstp(self, pil_image: Image.Image, detected_boxes: np.ndarray) -> int:
        """Select MSTP using the MSTP selector."""
        self.selector.eval()
        
        # Prepare candidate crops
        candidate_crops = []
        for box in detected_boxes:
            crop = pil_image.crop((box[0], box[1], box[2], box[3]))
            crop_tensor = self.crop_transform(crop)
            candidate_crops.append(crop_tensor)
        
        # Prepare global context
        global_tensor = self.global_transform(pil_image)
        
        # Stack candidates and run selector
        candidate_batch = torch.stack(candidate_crops).to(self.device)
        global_batch = global_tensor.to(self.device)
        
        with torch.no_grad():
            model_scores = self.selector(candidate_batch, global_batch).cpu().numpy()
        
        # Apply retrieval augmentation if enabled
        if self.use_retrieval and self.feature_bank is not None:
            retrieval_scores = self._compute_retrieval_scores(candidate_crops)
            final_scores = self.alpha * model_scores + (1 - self.alpha) * retrieval_scores
        else:
            final_scores = model_scores
        
        # Return index of highest scoring candidate
        return int(np.argmax(final_scores))
    
    def _compute_retrieval_scores(self, candidate_crops: List[torch.Tensor]) -> np.ndarray:
        """Compute retrieval scores for candidate regions."""
        retrieval_scores = []
        
        for crop_tensor in candidate_crops:
            # Convert tensor back to PIL for feature extraction
            crop_pil = T.ToPILImage()(crop_tensor.cpu())
            
            # Extract features
            features = self._extract_features(crop_pil)
            
            # Compute similarity with feature bank
            score = self.feature_bank.compute_similarity_score(features)
            retrieval_scores.append(score)
        
        return np.array(retrieval_scores)
    
    def _extract_features(self, pil_image: Image.Image) -> np.ndarray:
        """Extract features from image using pre-trained ResNet18."""
        self.feature_extractor.eval()
        
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])
        
        input_tensor = transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            features = features.view(-1).cpu().numpy()
        
        return features
    
    def _visualize_results(self, cv_image: np.ndarray, detected_boxes: np.ndarray, 
                          selected_index: int) -> np.ndarray:
        """Visualize detection and selection results."""
        result_image = cv_image.copy()
        
        for i, box in enumerate(detected_boxes):
            x1, y1, x2, y2 = box.astype(int)
            
            # Color: red for MSTP, green for other STPs
            color = (0, 0, 255) if i == selected_index else (0, 255, 0)
            label = "MSTP" if i == selected_index else "STP"
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            cv2.putText(result_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result_image
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         score_threshold: float = 0.7) -> Dict:
        """
        Process all images in a directory.
        
        Args:
            input_dir (str): Directory containing input images
            output_dir (str): Directory to save processed images
            score_threshold (float): Confidence threshold for detection
            
        Returns:
            Dict: Results summary for all processed images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for filename in sorted(os.listdir(input_dir)):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(input_dir, filename)
                
                # Process image
                processed_img, detected_boxes, selected_index = self.process_single_image(
                    image_path, score_threshold
                )
                
                if processed_img is not None:
                    # Save processed image
                    output_path = os.path.join(output_dir, filename)
                    cv2.imwrite(output_path, processed_img)
                    
                    # Store results
                    results[filename] = {
                        "candidate_boxes": detected_boxes.tolist(),
                        "predicted_MSTP_index": selected_index,
                        "num_candidates": len(detected_boxes)
                    }
                    
                    print(f"Processed: {filename}")
        
        # Save results summary
        results_path = os.path.join(output_dir, "inference_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_dir}")
        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Visual Guidance System Inference Pipeline"
    )
    
    # Input/output arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path to single image")
    input_group.add_argument("--img_dir", type=str, help="Path to input directory")
    
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    
    # Model arguments
    parser.add_argument("--detector_model_path", type=str, required=True,
                       help="Path to STP detector checkpoint")
    parser.add_argument("--selector_model_path", type=str, required=True,
                       help="Path to MSTP selector checkpoint")
    
    # Inference parameters
    parser.add_argument("--score_threshold", type=float, default=0.7,
                       help="Detection confidence threshold")
    parser.add_argument("--use_retrieval", action="store_true",
                       help="Enable retrieval augmentation")
    parser.add_argument("--retrieval_bank_file", type=str, default="feature_bank.pkl",
                       help="Path to feature bank file")
    parser.add_argument("--alpha", type=float, default=0.7,
                       help="Weight for retrieval vs model scores")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = VisualGuidancePipeline(
        detector_model_path=args.detector_model_path,
        selector_model_path=args.selector_model_path,
        use_retrieval=args.use_retrieval,
        retrieval_bank_file=args.retrieval_bank_file,
        alpha=args.alpha
    )
    
    # Process input
    if args.image:
        # Single image mode
        processed_img, detected_boxes, selected_index = pipeline.process_single_image(
            args.image, args.score_threshold
        )
        
        if processed_img is not None:
            cv2.imshow("Visual Guidance Result", processed_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        # Directory mode
        output_dir = args.output_dir or os.path.join(args.img_dir, "results")
        pipeline.process_directory(args.img_dir, output_dir, args.score_threshold)


if __name__ == "__main__":
    main()
