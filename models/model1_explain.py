#!/usr/bin/env python
"""
model1_explain.py

This script provides visual explanation methods for Model 1 (STP Detector).
It implements three visualization modes:

1. Grad-CAM++ Visualization:
   - Attaches forward and backward hooks to a target convolutional layer (e.g., model.backbone.body.layer4)
     to compute a Grad-CAM++ heatmap that highlights the key regions influencing a chosen detection.
   - The heatmap is overlaid on the original image.

2. Proposal Visualization:
   - Overlays predicted candidate boxes and the ground-truth boxes on the image,
     allowing comparison between the model's proposals and human annotations.

3. Occlusion Experiment:
   - Applies a sliding-window occlusion on the image and measures the drop in the top detection score.
   - Generates a heatmap indicating the sensitivity of the detection score to occlusion in different regions.

Usage:
    python model1_explain.py --mode gradcam --image 20250208194254_1.jpg --ann_file stp_labels.json --model_path model1_stp_detector.pth
    python model1_explain.py --mode proposal --image 20250208194254_1.jpg --ann_file stp_labels.json --model_path model1_stp_detector.pth
    python model1_explain.py --mode occlusion --image 20250208194254_1.jpg --model_path model1_stp_detector.pth --window_size 50 --stride 25
"""

import os
import cv2
import json
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

# ----------------------------------------------------------------------
# Global configuration dictionary: change these values as needed
# ----------------------------------------------------------------------
CONFIG = {
    "iou_threshold": 0.5,  # IoU threshold for evaluation
    "score_threshold": 0.5,  # Score threshold to filter predictions
}


# ----------------------------------------------------------------------
# Helper: Transformation pipeline (with default train flag)
# ----------------------------------------------------------------------
def get_transform(train=False):
    """
    Return a transformation pipeline.
    If train is True, include a random horizontal flip.
    """
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# ----------------------------------------------------------------------
# Load Model 1: STP Detector
# ----------------------------------------------------------------------
def get_model(num_classes=2):
    """
    Load a pre-trained Faster R-CNN model and modify its head for 2 classes.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ----------------------------------------------------------------------
# Grad-CAM++ Visualization Function
# ----------------------------------------------------------------------
def grad_cam_explain(model, image_tensor, target_layer, detection_index=0, score_threshold=0.5):
    """
    Perform a Grad-CAM++ explanation for a chosen detection.

    Parameters:
      model: The detection model.
      image_tensor: Input image tensor of shape [1, C, H, W] (with batch dimension).
      target_layer: The layer for which to extract activations and gradients.
      detection_index: The index of the detection (after filtering) to explain.
      score_threshold: Only consider predictions with a score >= this threshold.

    Returns:
      heatmap: A numpy array (H, W) representing the Grad-CAM++ heatmap.
    """
    # Containers to store activations and gradients.
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    # Register hooks on the target layer.
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    # Remove the extra batch dimension before passing the image to the model.
    image_single = image_tensor.squeeze(0)

    # Forward pass.
    model.eval()
    output = model([image_single])[0]
    scores = output["scores"].detach().cpu().numpy()
    boxes = output["boxes"].detach().cpu().numpy()
    valid_indices = np.where(scores >= score_threshold)[0]
    if len(valid_indices) == 0:
        print("No detection above the score threshold.")
        handle_forward.remove()
        handle_backward.remove()
        return None
    # Choose the detection_index (e.g., 0 for the highest-scoring detection).
    chosen_idx = valid_indices[detection_index]
    target_score = output["scores"][chosen_idx]

    # Backward pass: compute gradients of the target score with respect to the target layer.
    model.zero_grad()
    target_score.backward(retain_graph=True)

    # Remove hooks.
    handle_forward.remove()
    handle_backward.remove()

    # Retrieve activations and gradients.
    activations = activations[0]  # shape: (1, C, H', W')
    gradients = gradients[0]  # shape: (1, C, H', W')

    eps = 1e-6
    grad_squared = gradients ** 2
    grad_cubed = gradients ** 3
    sum_grad_cubed = torch.sum(activations * grad_cubed, dim=(2, 3), keepdim=True)
    denom = 2 * grad_squared + sum_grad_cubed
    alpha = grad_squared / (denom + eps)
    relu_gradients = F.relu(gradients)
    weights = torch.sum(alpha * relu_gradients, dim=(2, 3))  # shape: (1, C)
    activation_map = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * activations, dim=1)  # shape: (1, H', W')
    cam = F.relu(activation_map)
    cam = cam.squeeze(0).cpu().numpy()
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + eps)
    heatmap = cv2.resize(cam, (image_tensor.shape[3], image_tensor.shape[2]))
    return heatmap


# ----------------------------------------------------------------------
# Proposal Visualization Function
# ----------------------------------------------------------------------
def proposal_visualization(image_path, ann_file, model, device, score_threshold=0.5):
    """
    Visualize proposals by overlaying the predicted candidate boxes and the ground-truth boxes.
    Ground-truth boxes are drawn in blue; predicted boxes are drawn in green.
    """
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print(f"Error: Could not load image {image_path}")
        return
    image_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    transform = get_transform(train=False)
    img_tensor = transform(pil_img).to(device)

    # Load ground-truth annotation for this image.
    with open(ann_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    record = None
    for rec in annotations:
        if rec["image_id"] == os.path.basename(image_path):
            record = rec
            break
    if record is None:
        print(f"No annotation found for image {image_path}")
        return
    gt_boxes = []
    if "MSTP" in record and record["MSTP"]:
        gt_boxes.append(record["MSTP"])
    if "STP" in record and record["STP"]:
        gt_boxes.extend(record["STP"])

    model.eval()
    with torch.no_grad():
        output = model([img_tensor])[0]
    pred_boxes = output["boxes"].cpu().numpy().tolist()
    pred_scores = output["scores"].cpu().numpy().tolist()
    valid_idx = np.where(np.array(pred_scores) >= score_threshold)[0]
    pred_boxes = [pred_boxes[i] for i in valid_idx]

    # Draw ground-truth boxes in blue.
    for box in gt_boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(orig_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(orig_img, "GT", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    # Draw predicted boxes in green.
    for box in pred_boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(orig_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(orig_img, "Pred", (x_min, y_max + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Proposal Visualization", orig_img)
    print("Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ----------------------------------------------------------------------
# Occlusion Experiment Function
# ----------------------------------------------------------------------
def occlusion_experiment(model, image_tensor, device, window_size=50, stride=25, score_threshold=0.5):
    """
    Perform an occlusion experiment by sliding a window over the image and measuring the drop in
    the top detection score when a region is occluded.

    Parameters:
      model: The detection model.
      image_tensor: Input image tensor of shape [1, C, H, W].
      device: The torch device.
      window_size: The size of the occlusion window.
      stride: The stride with which to slide the window.
      score_threshold: The score threshold to consider a detection.

    Returns:
      heatmap: A numpy array of shape (H, W) where higher values indicate regions that,
               when occluded, cause a larger drop in detection score.
    """
    model.eval()
    image_single = image_tensor.squeeze(0)
    with torch.no_grad():
        output = model([image_single])[0]
    if len(output["scores"]) == 0:
        print("No detections found in the original image.")
        return None
    original_score = output["scores"].max().item()
    image_np = image_tensor.squeeze(0).cpu().numpy()  # shape: [C, H, W]
    C, H, W = image_np.shape
    heatmap = np.zeros((H, W), dtype=np.float32)
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            occluded = image_np.copy()
            mean_val = np.mean(image_np, axis=(1, 2), keepdims=True)
            occluded[:, y:min(y + window_size, H), x:min(x + window_size, W)] = mean_val
            occluded_tensor = torch.from_numpy(occluded).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model([occluded_tensor.squeeze(0)])[0]
            new_score = out["scores"].max().item() if len(out["scores"]) > 0 else 0.0
            score_drop = original_score - new_score
            heatmap[y:min(y + window_size, H), x:min(x + window_size, W)] = score_drop
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-6)
    return heatmap


# ----------------------------------------------------------------------
# Main function: Argument parsing and mode selection
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Model 1 Explanation (Interpretability Analysis)")
    parser.add_argument("--mode", type=str, choices=["gradcam", "proposal", "occlusion"], required=True,
                        help="Visualization mode: 'gradcam' for Grad-CAM++ heatmap, 'proposal' for proposal visualization, 'occlusion' for occlusion experiment")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the image for explanation")
    parser.add_argument("--ann_file", type=str, default="./stp_labels.json",
                        help="Annotation JSON file (used for proposal visualization)")
    parser.add_argument("--model_path", type=str, default="model1_stp_detector_best.pth",
                        help="Path to load the trained Model 1 weights")
    parser.add_argument("--score_threshold", type=float, default=0.5,
                        help="Score threshold for detection (default: 0.5)")
    parser.add_argument("--window_size", type=int, default=50,
                        help="Occlusion window size (default: 50)")
    parser.add_argument("--stride", type=int, default=25,
                        help="Stride for occlusion experiment (default: 25)")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Load the model.
    num_classes = 2
    model = get_model(num_classes)
    model.to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model weights from {args.model_path}")
    else:
        print(f"Model weights not found at {args.model_path}")
        return

    if args.mode == "gradcam":
        target_layer = model.backbone.body.layer4
        pil_img = Image.open(args.image).convert("RGB")
        transform = get_transform(train=False)
        img_tensor = transform(pil_img).unsqueeze(0).to(device)
        heatmap = grad_cam_explain(model, img_tensor, target_layer, detection_index=0,
                                   score_threshold=args.score_threshold)
        if heatmap is None:
            print("Grad-CAM++ failed to produce a heatmap.")
            return
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        orig_img = cv2.imread(args.image)
        overlay = cv2.addWeighted(orig_img, 0.5, heatmap_color, 0.5, 0)
        cv2.imshow("Grad-CAM++ Visualization", overlay)
        print("Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.mode == "proposal":
        proposal_visualization(args.image, args.ann_file, model, device, score_threshold=args.score_threshold)

    elif args.mode == "occlusion":
        pil_img = Image.open(args.image).convert("RGB")
        transform = get_transform(train=False)
        img_tensor = transform(pil_img).unsqueeze(0).to(device)
        heatmap = occlusion_experiment(model, img_tensor, device, window_size=args.window_size, stride=args.stride,
                                       score_threshold=args.score_threshold)
        if heatmap is None:
            print("Occlusion experiment failed to produce a heatmap.")
            return
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        orig_img = cv2.imread(args.image)
        overlay = cv2.addWeighted(orig_img, 0.5, heatmap_color, 0.5, 0)
        cv2.imshow("Occlusion Experiment", overlay)
        print("Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
