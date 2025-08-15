#!/usr/bin/env python
"""
model2_explain.py

This script provides visual explanation methods for Model 2 (MSTP Selector with Global Context).
Model 2 takes as input:
    - A list of candidate region crops extracted from an image.
    - A global context image (a low-resolution thumbnail of the entire image) as additional context.
The network fuses features from candidate crops (using a ResNet18 branch) with global context features
(from a lightweight CNN branch) to output a scalar score for each candidate.
Training is supervised via CrossEntropyLoss so that the candidate corresponding to the MSTP
(ground truth index) gets the highest score.

This script implements three explanation modes:
    1. Grad-CAM Visualization on a Candidate:
         - Registers forward and backward hooks on a target layer in the candidate branch to compute a Grad-CAM heatmap.
         - Overlays the heatmap on the selected candidate crop to show which regions are most influential.
    2. Occlusion Experiment on a Candidate:
         - Applies a sliding-window occlusion on a candidate crop and measures the drop in its predicted score.
         - Generates a sensitivity heatmap indicating which parts of the candidate are most critical.
    3. Comparison Visualization:
         - Processes all candidate crops for the image and computes their predicted scores.
         - Displays all candidate images in a grid with their scores annotated.
         - Highlights the candidate with the highest score (selected as MSTP) to allow comparison.

Usage:
    python model2_explain.py --mode gradcam --image 20250208193702_1.jpg --ann_file model2_test_dataset.json --model_path model2_mstp_selector.pth --candidate_index 0
    python model2_explain.py --mode occlusion --image 20250208193702_1.jpg --ann_file model2_test_dataset.json --model_path model2_mstp_selector.pth --candidate_index 0 --window_size 30 --stride 15
    python model2_explain.py --mode comparison --image 20250208193702_1.jpg --ann_file model2_test_dataset.json --model_path model2_mstp_selector.pth
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import json
import argparse
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as T
import torchvision.models as models
import matplotlib.pyplot as plt


# -------------------------
# Custom collate function (for variable-length candidate lists)
# -------------------------
def selector_collate_fn(batch):
    # Each sample is a dictionary; simply return the list of samples.
    return batch


# -------------------------
# Dataset for Model 2 with Global Context
# -------------------------
class MSTPSelectorDataset(torch.utils.data.Dataset):
    """
    This dataset reads from the prepared JSON file (e.g., "model2_dataset.json").
    For each record, it loads the image from the specified directory,
    extracts candidate crops (by cropping using candidate bounding boxes),
    and also extracts a global context image as a low-resolution thumbnail.
    The candidate crops are randomly shuffled and the ground-truth index is updated accordingly.

    The returned dictionary has the following keys:
       "candidates": list of candidate crop tensors (each of shape [3, 224, 224])
       "global": global context image tensor (shape [3, 64, 64])
       "gt_index": updated ground-truth index after shuffling
       "image_id": the filename of the image
    """

    def __init__(self, annotations_file, img_dir, crop_transform=None, global_transform=None):
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.records = json.load(f)
        self.img_dir = img_dir
        self.crop_transform = crop_transform if crop_transform is not None else T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])
        self.global_transform = global_transform if global_transform is not None else T.Compose([
            T.Resize((64, 64)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image_path = os.path.join(self.img_dir, rec["image_id"])
        image = Image.open(image_path).convert("RGB")
        # If "candidates" key is missing, generate it from MSTP and STP.
        if "candidates" not in rec:
            candidates = []
            if "MSTP" in rec and rec["MSTP"]:
                candidates.append(rec["MSTP"])
            if "STP" in rec and rec["STP"]:
                candidates.extend(rec["STP"])
            rec["candidates"] = candidates
        candidate_boxes = rec["candidates"]
        if len(candidate_boxes) == 0:
            return {"candidates": [], "global": None, "gt_index": -1, "image_id": rec["image_id"]}
        candidates = []
        for box in candidate_boxes:
            crop = image.crop((box[0], box[1], box[2], box[3]))
            crop = self.crop_transform(crop)
            candidates.append(crop)
        global_img = self.global_transform(image)
        gt_index = rec["gt_index"]
        indices = list(range(len(candidates)))
        random.shuffle(indices)
        shuffled_candidates = [candidates[i] for i in indices]
        new_gt_index = indices.index(gt_index)
        return {"candidates": shuffled_candidates, "global": global_img, "gt_index": new_gt_index,
                "image_id": rec["image_id"]}


# -------------------------
# MSTP Selector Network with Global Context Fusion
# -------------------------
class MSTPSelectorNet(nn.Module):
    def __init__(self):
        super(MSTPSelectorNet, self).__init__()
        # Candidate branch: pre-trained ResNet18 (exclude final FC)
        resnet = models.resnet18(pretrained=True)
        candidate_modules = list(resnet.children())[:-1]  # Remove final FC
        # Remove the global average pooling to obtain spatial feature maps.
        self.candidate_branch = nn.Sequential(*candidate_modules[:-1])
        # Global context branch: a lightweight CNN for low-resolution image.
        self.global_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 16 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 32 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64 x 8 x 8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 64 x 1 x 1
            nn.Flatten(),  # 64
            nn.Linear(64, 512),
            nn.ReLU()
        )
        # Final classifier: fuse candidate feature (512-d) and global context feature (512-d)
        self.classifier = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, candidate_imgs, global_img):
        """
        candidate_imgs: tensor of shape (N, 3, 224, 224) for N candidate crops.
        global_img: tensor of shape (3, 64, 64) representing the global context.
        """
        # Extract candidate features.
        candidate_feats = self.candidate_branch(candidate_imgs)  # (N, 512, H', W')
        candidate_feats = torch.nn.functional.adaptive_avg_pool2d(candidate_feats, (1, 1))
        candidate_feats = candidate_feats.view(candidate_feats.size(0), -1)  # (N, 512)
        # Extract global context feature.
        global_feat = self.global_branch(global_img.unsqueeze(0))  # (1, 512)
        global_feat_expanded = global_feat.expand(candidate_feats.size(0), -1)  # (N, 512)
        fused_feats = torch.cat([candidate_feats, global_feat_expanded], dim=1)  # (N, 1024)
        scores = self.classifier(fused_feats)  # (N, 1)
        scores = scores.squeeze(1)  # (N,)
        return scores


# ----------------------------------------------------------------------
# Grad-CAM Visualization for a Candidate in Model 2
# ----------------------------------------------------------------------
def grad_cam_candidate(model, candidate_img_tensor, global_img_tensor, target_layer, candidate_index=0):
    """
    Compute Grad-CAM for a selected candidate crop using the candidate branch.

    Parameters:
      model: The MSTP Selector model.
      candidate_img_tensor: Tensor of shape [1, 3, 224, 224] representing one candidate crop.
      global_img_tensor: Tensor of shape [3, 64, 64] representing the global context.
      target_layer: The layer in the candidate branch on which to register hooks.
      candidate_index: The index of the candidate to explain (should be 0 when a single candidate is passed).

    Returns:
      heatmap: A numpy array of shape (H, W) representing the Grad-CAM heatmap for the candidate.
    """
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    model.eval()
    output = model(candidate_img_tensor, global_img_tensor)
    target_score = output[candidate_index]

    model.zero_grad()
    target_score.backward(retain_graph=True)

    handle_forward.remove()
    handle_backward.remove()

    act = activations[0]  # shape: [1, C, H', W']
    grad = gradients[0]  # shape: [1, C, H', W']
    eps = 1e-6
    grad_squared = grad ** 2
    grad_cubed = grad ** 3
    sum_grad_cubed = torch.sum(act * grad_cubed, dim=(2, 3), keepdim=True)
    denom = 2 * grad_squared + sum_grad_cubed
    alpha = grad_squared / (denom + eps)
    relu_grad = F.relu(grad)
    weights = torch.sum(alpha * relu_grad, dim=(2, 3))  # shape: [1, C]
    cam = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * act, dim=1)  # shape: [1, H', W']
    cam = F.relu(cam)
    cam = cam.squeeze(0).cpu().numpy()
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + eps)
    heatmap = cv2.resize(cam, (candidate_img_tensor.shape[3], candidate_img_tensor.shape[2]))
    return heatmap


# ----------------------------------------------------------------------
# Occlusion Experiment for a Candidate in Model 2
# ----------------------------------------------------------------------
def occlusion_experiment_candidate(model, candidate_img_tensor, global_img_tensor, device, window_size=30, stride=15):
    """
    Perform an occlusion experiment on a candidate crop.
    Occlude regions in the candidate image and measure the drop in its predicted score.

    Parameters:
      model: The MSTP Selector model.
      candidate_img_tensor: Tensor of shape [1, 3, 224, 224] representing the candidate crop.
      global_img_tensor: Tensor of shape [3, 64, 64] representing the global context.
      device: Torch device.
      window_size: The occlusion window size.
      stride: The sliding window stride.

    Returns:
      heatmap: A numpy array (H, W) showing the score drop for each occlusion region.
    """
    model.eval()
    with torch.no_grad():
        original_output = model(candidate_img_tensor, global_img_tensor)
    original_score = original_output.max().item()

    candidate_np = candidate_img_tensor.squeeze(0).cpu().numpy()  # shape: [3, 224, 224]
    C, H, W = candidate_np.shape
    heatmap = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            occluded = candidate_np.copy()
            mean_val = np.mean(candidate_np, axis=(1, 2), keepdims=True)
            occluded[:, y:min(y + window_size, H), x:min(x + window_size, W)] = mean_val
            occluded_tensor = torch.from_numpy(occluded).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(occluded_tensor, global_img_tensor)
            new_score = out.max().item() if out.numel() > 0 else 0.0
            score_drop = original_score - new_score
            heatmap[y:min(y + window_size, H), x:min(x + window_size, W)] = score_drop
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-6)
    return heatmap


# ----------------------------------------------------------------------
# Comparison Visualization for All Candidate Crops
# ----------------------------------------------------------------------
def comparison_visualization(image_path, ann_file, model, device, crop_transform, global_transform):
    """
    Compare all candidate crops by computing and displaying their predicted scores.
    This function extracts all candidate crops from the image (using the annotation record),
    computes the score for each candidate using Model 2, and then displays a grid of the candidate images
    along with their predicted scores. The candidate with the highest score (i.e., selected as MSTP)
    is highlighted.
    """
    # Load annotation record.
    with open(ann_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    image_filename = os.path.basename(image_path)
    record = None
    for rec in records:
        if rec["image_id"] == image_filename:
            record = rec
            break
    if record is None:
        print(f"Record not found for image {image_filename}")
        return

    # Load full image.
    image = Image.open(image_path).convert("RGB")
    # Extract candidate boxes.
    candidate_boxes = record["candidates"]
    if len(candidate_boxes) == 0:
        print("No candidate boxes for this image.")
        return

    candidates = []
    for box in candidate_boxes:
        crop = image.crop((box[0], box[1], box[2], box[3]))
        crop = crop_transform(crop)
        candidates.append(crop)

    # Extract global context image.
    global_img = global_transform(image)
    # Stack all candidate crops.
    candidate_tensor = torch.stack(candidates).to(device)  # shape: (N, 3, 224, 224)
    global_tensor = global_img.to(device)

    model.eval()
    with torch.no_grad():
        scores = model(candidate_tensor, global_tensor)  # shape: (N,)
    scores_np = scores.cpu().numpy()
    # Find the index of the highest scoring candidate.
    best_idx = int(np.argmax(scores_np))

    # Create a figure showing each candidate with its score.
    import matplotlib.pyplot as plt
    N = candidate_tensor.size(0)
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))
    plt.figure(figsize=(15, 15))
    for i in range(N):
        candidate_np = candidate_tensor[i].cpu().numpy().transpose(1, 2, 0)
        candidate_np = np.clip(candidate_np * 255, 0, 255).astype(np.uint8)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(candidate_np)
        title_str = f"Score: {scores_np[i]:.3f}"
        if i == best_idx:
            title_str += " <-- MSTP"
        plt.title(title_str, color="red" if i == best_idx else "green")
        plt.axis("off")
    plt.suptitle("Candidate Comparison Visualization", fontsize=16)
    plt.show()


# ----------------------------------------------------------------------
# Main function: Argument parsing and mode selection for Model 2 explanation
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Model 2 Explanation (Interpretability Analysis for MSTP Selector)")
    parser.add_argument("--mode", type=str, choices=["gradcam", "occlusion", "comparison"], required=True,
                        help="Explanation mode: 'gradcam' for candidate Grad-CAM visualization, 'occlusion' for occlusion experiment on candidate crop, 'comparison' for comparing all candidate scores")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the image for explanation")
    parser.add_argument("--ann_file", type=str, default="model2_dataset.json",
                        help="Path to the prepared dataset JSON (for candidate extraction)")
    parser.add_argument("--model_path", type=str, default="model2_mstp_selector.pth",
                        help="Path to load the trained Model 2 weights")
    parser.add_argument("--candidate_index", type=int, default=0,
                        help="Index of the candidate crop to analyze (default: 0)")
    parser.add_argument("--window_size", type=int, default=30,
                        help="Occlusion window size (default: 30)")
    parser.add_argument("--stride", type=int, default=15,
                        help="Occlusion stride (default: 15)")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Define transforms for candidate crop and global context.
    crop_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    global_transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor()
    ])

    # Load the model.
    model = MSTPSelectorNet()
    model.to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model weights from {args.model_path}")
    else:
        print(f"Model weights not found at {args.model_path}")
        return

    # Load the JSON record for the given image.
    with open(args.ann_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    image_filename = os.path.basename(args.image)
    record = None
    for rec in records:
        if rec["image_id"] == image_filename:
            record = rec
            break
    if record is None:
        print(f"Record not found for image {image_filename}")
        return
    # If "candidates" key is missing, generate it.
    if "candidates" not in record:
        candidates = []
        if "MSTP" in record and record["MSTP"]:
            candidates.append(record["MSTP"])
        if "STP" in record and record["STP"]:
            candidates.extend(record["STP"])
        record["candidates"] = candidates

    # Load the full image.
    image = Image.open(args.image).convert("RGB")
    # Extract candidate crops.
    candidate_boxes = record["candidates"]
    if len(candidate_boxes) == 0:
        print("No candidate boxes for this image.")
        return
    candidates_list = []
    for box in candidate_boxes:
        crop = image.crop((box[0], box[1], box[2], box[3]))
        crop = crop_transform(crop)
        candidates_list.append(crop)
    # For gradcam and occlusion modes, select one candidate using candidate_index.
    if args.mode in ["gradcam", "occlusion"]:
        try:
            candidate_img_tensor = torch.stack([candidates_list[args.candidate_index]]).to(device)
        except IndexError:
            print("Candidate index out of range.")
            return
    # Extract global context image.
    global_img_tensor = global_transform(image).to(device)

    if args.mode == "gradcam":
        # For Grad-CAM, register hooks on a target layer in the candidate branch.
        # Here we choose the last convolutional layer of the candidate branch.
        target_layer = model.candidate_branch[-1]
        heatmap = grad_cam_candidate(model, candidate_img_tensor, global_img_tensor, target_layer, candidate_index=0)
        if heatmap is None:
            print("Grad-CAM failed to produce a heatmap.")
            return
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        candidate_np = candidate_img_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        candidate_np = (candidate_np * 255).astype(np.uint8)
        overlay = cv2.addWeighted(candidate_np, 0.5, heatmap_color, 0.5, 0)
        cv2.imshow("Candidate Grad-CAM Visualization", overlay)
        print("Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.mode == "occlusion":
        heatmap = occlusion_experiment_candidate(model, candidate_img_tensor, global_img_tensor, device,
                                                 window_size=args.window_size, stride=args.stride)
        if heatmap is None:
            print("Occlusion experiment failed to produce a heatmap.")
            return
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        candidate_np = candidate_img_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        candidate_np = (candidate_np * 255).astype(np.uint8)
        overlay = cv2.addWeighted(candidate_np, 0.5, heatmap_color, 0.5, 0)
        cv2.imshow("Candidate Occlusion Visualization", overlay)
        print("Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.mode == "comparison":
        # In comparison mode, process all candidate crops and compute their scores.
        candidate_tensor = torch.stack(candidates_list).to(device)  # shape: (N, 3, 224, 224)
        global_tensor = global_img_tensor  # shape: (3, 64, 64)
        model.eval()
        with torch.no_grad():
            scores = model(candidate_tensor, global_tensor)  # shape: (N,)
        scores_np = scores.cpu().numpy()
        best_idx = int(np.argmax(scores_np))
        # Display candidate images in a grid with their scores.
        import matplotlib.pyplot as plt
        N = candidate_tensor.size(0)
        cols = int(np.ceil(np.sqrt(N)))
        rows = int(np.ceil(N / cols))
        plt.figure(figsize=(15, 15))
        for i in range(N):
            candidate_np = candidate_tensor[i].cpu().numpy().transpose(1, 2, 0)
            candidate_np = np.clip(candidate_np * 255, 0, 255).astype(np.uint8)
            plt.subplot(rows, cols, i + 1)
            title_str = f"Score: {scores_np[i]:.3f}"
            if i == best_idx:
                title_str += " <-- MSTP"
                plt.title(title_str, color="red")
            else:
                plt.title(title_str, color="green")
            plt.imshow(candidate_np)
            plt.axis("off")
        plt.suptitle("Candidate Comparison Visualization", fontsize=16)
        plt.show()


if __name__ == "__main__":
    main()
