#!/usr/bin/env python
"""
Demo script for Visual Guidance System

This script demonstrates how to use the complete Visual Guidance System
for STP detection and MSTP selection on sample images.
"""

import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image

# Fix import path for direct script execution
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Use absolute imports instead of relative imports
from pipeline.inference import VisualGuidancePipeline
from config.config import Config


def demo_single_image(pipeline: VisualGuidancePipeline, image_path: str, 
                      output_path: str = None, show_result: bool = True):
    """
    Demonstrate the system on a single image.
    
    Args:
        pipeline (VisualGuidancePipeline): Initialized pipeline
        image_path (str): Path to input image
        output_path (str): Path to save output image (optional)
        show_result (bool): Whether to display the result
    """
    print(f"\nProcessing image: {image_path}")
    
    # Process image
    processed_img, detected_boxes, selected_index = pipeline.process_single_image(
        image_path, score_threshold=0.7
    )
    
    if processed_img is None:
        print("No STPs detected in the image.")
        return
    
    # Display results
    print(f"Detected {len(detected_boxes)} STP candidates")
    print(f"Selected MSTP index: {selected_index}")
    
    # Show bounding boxes
    for i, box in enumerate(detected_boxes):
        x1, y1, x2, y2 = box.astype(int)
        label = "MSTP" if i == selected_index else "STP"
        print(f"  {label} {i}: [{x1}, {y1}, {x2}, {y2}]")
    
    # Save output if specified
    if output_path:
        cv2.imwrite(output_path, processed_img)
        print(f"Result saved to: {output_path}")
    
    # Display result
    if show_result:
        cv2.imshow("Visual Guidance Result", processed_img)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def demo_directory(pipeline: VisualGuidancePipeline, input_dir: str, 
                   output_dir: str, max_images: int = 5):
    """
    Demonstrate the system on a directory of images.
    
    Args:
        pipeline (VisualGuidancePipeline): Initialized pipeline
        input_dir (str): Input directory containing images
        output_dir (str): Output directory for results
        max_images (int): Maximum number of images to process
    """
    print(f"\nProcessing directory: {input_dir}")
    
    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [
        f for f in os.listdir(input_dir) 
        if any(f.lower().endswith(ext) for ext in image_extensions)
    ]
    
    if not image_files:
        print("No image files found in the directory.")
        return
    
    # Limit number of images for demo
    image_files = image_files[:max_images]
    print(f"Processing {len(image_files)} images...")
    
    # Process images
    results = pipeline.process_directory(input_dir, output_dir, score_threshold=0.7)
    
    # Print summary
    print(f"\nProcessing completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Processed {len(results)} images")
    
    # Show sample results
    for i, (filename, result) in enumerate(list(results.items())[:3]):
        print(f"\nSample result {i+1}: {filename}")
        print(f"  Candidates: {result['num_candidates']}")
        print(f"  MSTP index: {result['predicted_MSTP_index']}")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Visual Guidance System Demo")
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path to single image")
    input_group.add_argument("--dir", type=str, help="Path to input directory")
    
    # Output arguments
    parser.add_argument("--output", type=str, help="Output path for results")
    
    # Model arguments
    parser.add_argument("--detector_model", type=str, required=True,
                       help="Path to STP detector checkpoint")
    parser.add_argument("--selector_model", type=str, required=True,
                       help="Path to MSTP selector checkpoint")
    
    # Pipeline arguments
    parser.add_argument("--use_retrieval", action="store_true",
                       help="Enable retrieval augmentation")
    parser.add_argument("--feature_bank", type=str, default="feature_bank.pkl",
                       help="Path to feature bank file")
    parser.add_argument("--alpha", type=float, default=0.7,
                       help="Weight for retrieval vs model scores")
    
    # Demo arguments
    parser.add_argument("--max_images", type=int, default=5,
                       help="Maximum images to process in directory mode")
    parser.add_argument("--no_display", action="store_true",
                       help="Disable result display")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Visual Guidance System Demo")
    print("=" * 60)
    
    # Check if models exist
    if not os.path.exists(args.detector_model):
        print(f"Error: STP detector model not found: {args.detector_model}")
        return
    
    if not os.path.exists(args.selector_model):
        print(f"Error: MSTP selector model not found: {args.selector_model}")
        return
    
    # Check feature bank if retrieval is enabled
    if args.use_retrieval and not os.path.exists(args.feature_bank):
        print(f"Warning: Feature bank not found: {args.feature_bank}")
        print("Retrieval augmentation will be disabled.")
        args.use_retrieval = False
    
    # Create pipeline
    print("\nInitializing Visual Guidance Pipeline...")
    try:
        pipeline = VisualGuidancePipeline(
            detector_model_path=args.detector_model,
            selector_model_path=args.selector_model,
            use_retrieval=args.use_retrieval,
            retrieval_bank_file=args.feature_bank,
            alpha=args.alpha
        )
        print("Pipeline initialized successfully!")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return
    
    # Run demo
    if args.image:
        # Single image mode
        output_path = args.output if args.output else None
        show_result = not args.no_display
        
        demo_single_image(pipeline, args.image, output_path, show_result)
        
    else:
        # Directory mode
        output_dir = args.output if args.output else os.path.join(args.dir, "demo_results")
        
        demo_directory(pipeline, args.dir, output_dir, args.max_images)
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()
