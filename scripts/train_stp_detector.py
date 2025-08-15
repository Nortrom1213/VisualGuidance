#!/usr/bin/env python
"""
Training script for STP Detector

This script trains the Spatial Transition Point (STP) detector using
the Faster R-CNN architecture with adapter modules for efficient
fine-tuning.
"""

import os
import sys
import argparse
import json
import logging
from typing import Dict, List, Tuple

# Fix import path for direct script execution
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

# Use absolute imports instead of relative imports
from models.stp_detector import STPDetector, STPDataset, collate_fn, get_stp_detector
from config.config import Config


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_data_loaders(train_dataset: STPDataset, val_dataset: STPDataset,
                        batch_size: int, num_workers: int) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create training and validation data loaders."""
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_one_epoch(model: STPDetector, data_loader: torch.utils.data.DataLoader,
                    optimizer: optim.Optimizer, device: torch.device,
                    epoch: int, logger: logging.Logger) -> float:
    """Train the model for one epoch."""
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, targets) in enumerate(data_loader):
        try:
            # Move data to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass - ensure model is in train mode
            model.train()
            loss_dict = model(images, targets)
            
            # Check if loss_dict is valid
            if not isinstance(loss_dict, dict):
                logger.warning(f"Unexpected model output type in training: {type(loss_dict)}")
                continue
                
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(data_loader)}, Loss: {losses.item():.4f}")
                
        except Exception as e:
            logger.warning(f"Error during training batch {batch_idx}: {e}")
            continue
    
    if num_batches == 0:
        logger.warning("No valid training batches processed")
        return float('inf')
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate_model(model: STPDetector, data_loader: torch.utils.data.DataLoader,
                   device: torch.device, logger: logging.Logger) -> Dict[str, float]:
    """Validate the model on validation set."""
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            # Move data to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass - during validation, we need to temporarily set model to train mode
            # to get loss computation, then switch back to eval mode
            model.train()
            try:
                loss_dict = model(images, targets)
                if isinstance(loss_dict, dict):
                    losses = sum(loss for loss in loss_dict.values())
                    total_loss += losses.item()
                    num_batches += 1
                else:
                    logger.warning(f"Unexpected model output type: {type(loss_dict)}")
                    continue
            except Exception as e:
                logger.warning(f"Error during validation batch: {e}")
                continue
            finally:
                model.eval()  # Always restore eval mode
    
    if num_batches == 0:
        logger.warning("No valid validation batches processed")
        return {"val_loss": float('inf')}
    
    avg_loss = total_loss / num_batches
    
    # Note: For object detection, you might want to compute mAP here
    # This is a simplified validation that only computes loss
    
    return {"val_loss": avg_loss}


def save_checkpoint(model: STPDetector, optimizer: optim.Optimizer, epoch: int,
                   loss: float, save_path: str):
    """Save model checkpoint."""
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train STP Detector")
    
    # Data arguments
    parser.add_argument("--annotations_file", type=str, required=True,
                       help="Path to annotations JSON file")
    parser.add_argument("--images_dir", type=str, required=True,
                       help="Directory containing images")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory for checkpoints and logs")
    parser.add_argument("--num_epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=0.005,
                       help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
                       help="Momentum for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0005,
                       help="Weight decay")
    
    # Model arguments
    parser.add_argument("--use_adapter", action="store_true",
                       help="Use adapter modules for fine-tuning")
    parser.add_argument("--adapter_dim", type=int, default=256,
                       help="Adapter bottleneck dimension")
    
    # System arguments
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(log_dir)
    logger.info("Starting STP Detector training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create datasets
    logger.info("Creating datasets...")
    
    # Load annotations and split into train/val
    with open(args.annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Simple train/val split
    split_idx = int(len(annotations) * Config.STP_DETECTOR_CONFIG["train_val_split"])
    train_annotations = annotations[:split_idx]
    val_annotations = annotations[split_idx:]
    
    logger.info(f"Train samples: {len(train_annotations)}")
    logger.info(f"Val samples: {len(val_annotations)}")
    
    # Create datasets
    train_dataset = STPDataset(train_annotations, args.images_dir)
    val_dataset = STPDataset(val_annotations, args.images_dir)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset, args.batch_size, args.num_workers
    )
    
    # Create model
    logger.info("Creating model...")
    model = get_stp_detector(
        num_classes=2,
        adapter_dim=args.adapter_dim,
        use_adapter=args.use_adapter
    )
    model.to(device)
    
    # Create optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # Training
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, logger)
        
        # Validation
        val_metrics = validate_model(model, val_loader, device, logger)
        val_loss = val_metrics["val_loss"]
        
        # Learning rate scheduling
        scheduler.step()
        
        # Logging
        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Tensorboard logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, val_loss, best_checkpoint_path)
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
    
    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pth")
    save_checkpoint(model, optimizer, args.num_epochs - 1, val_loss, final_checkpoint_path)
    
    logger.info("Training completed!")
    writer.close()


if __name__ == "__main__":
    main()
