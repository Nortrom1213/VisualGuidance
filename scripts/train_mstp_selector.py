#!/usr/bin/env python
"""
Training script for MSTP Selector

This script trains the Main Spatial Transition Point (MSTP) selector
using a dual-branch architecture that combines local candidate features
with global context information.
"""

import os
import argparse
import json
import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from ..models.mstp_selector import (
    MSTPSelectorNet, MSTPSelectorDataset, selector_collate_fn,
    train_one_epoch, evaluate_selector, create_mstp_selector
)
from ..config.config import Config


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


def create_data_loaders(train_dataset: MSTPSelectorDataset, val_dataset: MSTPSelectorDataset,
                        batch_size: int, num_workers: int) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create training and validation data loaders."""
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=selector_collate_fn,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=selector_collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


def save_checkpoint(model: MSTPSelectorNet, optimizer: optim.Optimizer, epoch: int,
                   loss: float, accuracy: float, save_path: str):
    """Save model checkpoint."""
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train MSTP Selector")
    
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
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001,
                       help="Weight decay")
    
    # Model arguments
    parser.add_argument("--use_adapter", action="store_true",
                       help="Use adapter modules for fine-tuning")
    parser.add_argument("--bottleneck_dim", type=int, default=256,
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
    logger.info("Starting MSTP Selector training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create datasets
    logger.info("Creating datasets...")
    
    # Load annotations and split into train/val
    with open(args.annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Simple train/val split
    split_idx = int(len(annotations) * Config.DATASET_CONFIG["train_val_split"])
    train_annotations = annotations[:split_idx]
    val_annotations = annotations[split_idx:]
    
    logger.info(f"Train samples: {len(train_annotations)}")
    logger.info(f"Val samples: {len(val_annotations)}")
    
    # Create datasets
    train_dataset = MSTPSelectorDataset(train_annotations, args.images_dir)
    val_dataset = MSTPSelectorDataset(val_annotations, args.images_dir)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset, args.batch_size, args.num_workers
    )
    
    # Create model
    logger.info("Creating model...")
    model = create_mstp_selector(
        bottleneck_dim=args.bottleneck_dim,
        use_adapter=args.use_adapter
    )
    model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    logger.info("Starting training...")
    best_val_accuracy = 0.0
    
    for epoch in range(args.num_epochs):
        # Training
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, criterion
        )
        
        # Validation
        val_metrics = evaluate_selector(model, val_loader, device)
        val_accuracy = val_metrics["accuracy"]
        
        # Learning rate scheduling
        scheduler.step()
        
        # Logging
        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Tensorboard logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, train_loss, val_accuracy, best_checkpoint_path)
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch, train_loss, val_accuracy, checkpoint_path)
    
    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pth")
    save_checkpoint(model, optimizer, args.num_epochs - 1, train_loss, val_accuracy, final_checkpoint_path)
    
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
