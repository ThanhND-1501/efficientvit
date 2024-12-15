# File: train_deeplabv3_mobilevit_cityscapes.py
import argparse
import os
import wandb

from utils import *
from efficientvit.apps.utils import AverageMeter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Cityscapes
from transformers import MobileViTConfig, MobileViTForSemanticSegmentation
import numpy as np
from PIL import Image

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepLabV3-MobileViT on Cityscapes")
    parser.add_argument("--data_path", type=str, required=True, help="Path to Cityscapes dataset")
    parser.add_argument("--model_type", type=str, default="xx", choices=["x", "xx"], required=True, help="Type of MobileViT model.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save models")
    parser.add_argument("--save_interval", type=int, default=5, help="Interval (in epochs) to save the model")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume a checkpoint")
    parser.add_argument("--lr", type=float, default=0.0006, help="Learning rate for training")
    return parser.parse_args()

# Label remapping function
def remap_labels(target):
    """
    Remap labels using a predefined label map.
    """
    label_map = np.array(
        (
            16, 16, 16, 16, 16, 16, 16,
            0,  # road 7
            1,  # sidewalk 8
            16, 16,
            2,  # building 11
            2,  # wall 12
            2,  # fence 13
            16, 16, 16,
            3,  # pole 17
            16,
            4,  # traffic light 19
            5,  # traffic sign 20
            6,  # vegetation 21
            6,  # terrain 22
            7,  # sky 23
            8,  # person 24
            9,  # rider 25
            10,  # car 26
            11,  # truck 27
            12,  # bus 28
            16, 16,
            13,  # train 31
            14,  # motorcycle 32
            15,  # bicycle 33
        )
    )
    target = np.array(target)  # Convert target to numpy array
    target = label_map[target]  # Apply label map
    return torch.tensor(target, dtype=torch.long)

# Main function
def main():
    args = parse_args()

    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 17  # Based on the label_map (0-15)

    # Transforms for the dataset
    transform = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Target transform with label remapping
    def target_transform(target):
        return remap_labels(target)

    # Load Cityscapes dataset
    train_dataset = Cityscapes(
        root=args.data_path,
        split="train",
        mode="fine",
        target_type="semantic",
        transform=transform,
        target_transform=target_transform
    )

    val_dataset = Cityscapes(
        root=args.data_path,
        split="val",
        mode="fine",
        target_type="semantic",
        transform=transform,
        target_transform=target_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load the pretrained MobileViT-Small model from HuggingFace
    config = MobileViTConfig.from_pretrained(f"apple/deeplabv3-mobilevit-{args.model_type}-small")
    model = MobileViTForSemanticSegmentation.from_pretrained(
        f"apple/deeplabv3-mobilevit-{args.model_type}-small",
        config=config
    )
    model.to(DEVICE)

    # Update classifier head for the new label map
    if model.config.num_labels != NUM_CLASSES:
        model.segmentation_head.classifier = nn.Conv2d(
            in_channels=model.segmentation_head.classifier.convolution.in_channels,
            out_channels=NUM_CLASSES,
            kernel_size=1
        ).to(DEVICE)
        model.config.num_labels = NUM_CLASSES

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model.load_state_dict(torch.load(args.resume))

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Ignore remapped "void" class
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3, verbose=True)

    # Training and Validation functions
    def train_one_epoch(model, loader, optimizer, criterion, device):
        model.train()
        total_loss = 0
        for images, targets in loader:
            images, targets = images.to(device), targets.squeeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def validate(model, loader, criterion, device, interaction, union, iou):
        model.eval()
        total_loss = 0
        val_accs = []
        with torch.no_grad():
            for images, targets in loader:
                images, targets = images.to(device), targets.squeeze(1).to(device)
                raw_outputs = model(pixel_values=images)
                outputs = raw_outputs.logits
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                preds = raw_outputs.argmax(dim=1)
                stats = iou(preds, targets)
                interaction.update(stats["i"])
                union.update(stats["u"])
                
                acc = compute_metrics(preds.cpu().numpy(), targets.cpu().numpy(), NUM_CLASSES)
                val_accs.append(acc)
                
        return total_loss/len(loader), sum(val_accs)/len(val_accs)
    
    wandb.init(
        project="semantic-segmentation",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "save_interval": args.save_interval,
            "dataset": args.data_path,
        },
    )

    # Training Loop
    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0
    best_ckpt_path = ''
    last_ckpt_path = ''
    os.makedirs(os.path.join(args.save_dir, 'ckpt_interval'), exist_ok=True)
    
    for epoch in range(args.epochs):
        interaction = AverageMeter(is_distributed=False)
        union = AverageMeter(is_distributed=False)
        iou = SegIOU(17, ignore_index=16)
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE, interaction, union, iou)
        val_iou = (interaction.sum / union.sum).cpu().mean().item() * 100
        
        scheduler.step(val_loss)
        print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_accuracy": val_acc
        })
        
        # Save model checkpoint
        if best_acc < val_acc:
            best_acc = val_acc
            if os.path.exists(best_ckpt_path):
                os.remove(best_ckpt_path)
            best_ckpt_path = os.path.join(args.save_dir, f"best_model_epoch_{epoch}_iou_{val_iou}_acc_{val_acc}.pth")
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"Best model saved at {best_ckpt_path}")
        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, 'ckpt_interval', f"model_epoch_{epoch}_iou_{val_iou}_acc_{val_acc}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model periodically saved at {checkpoint_path}")
        if os.path.exists(last_ckpt_path):
            os.remove(last_ckpt_path)
        last_ckpt_path = os.path.join(args.save_dir, f"last_model_epoch_{epoch}_iou_{val_iou}_acc_{val_acc}.pth")
        torch.save(model.state_dict(), last_ckpt_path)
    wandb.finish()
    
if __name__ == "__main__":
    main()
