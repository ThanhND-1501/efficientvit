# !python train_efficientvit_seg_cityscapes.py --data_path --epochs --batch_size --save_dir --save_interval --resume --lr
import os
import argparse
import wandb

from cityscapes_pt import *
from utils import *
from efficientvit.apps.utils import AverageMeter
from efficientvit.seg_model_zoo import create_seg_model

from torch.utils.data import DataLoader
from torch import nn
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.datasets import Cityscapes
from tqdm import tqdm

# Define command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--model_type", type=str, default="b1", choices=["b0", "b1", "b2"], required=True, help="Type of EfficientViT model.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--save_interval", type=int, default=5, help="Interval (in epochs) to save checkpoints.")
    parser.add_argument("--resume", type=str, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--lr", type=float, default=0.0006, help="Learning rate for the optimizer.")
    parser.add_argument("--early_stop", type=int, default=10, help="Number of epochs to stop when there is no improvement in accuracy.")
    return parser.parse_args()

# Label remapping function
def remap_labels(target):
    """
    Remap labels using a predefined label map.
    """
    label_map = np.array(
        (
            15, 15, 15, 15, 15, 15, 15,
            0,  # road 7
            1,  # sidewalk 8
            15, 15,
            2,  # building 11
            2,  # wall 12
            2,  # fence 13
            15, 15, 15,
            3,  # pole 17
            15,
            4,  # traffic light 19
            5,  # traffic sign 20
            6,  # vegetation 21
            6,  # terrain 22
            7,  # sky 23
            8,  # person 24
            9,  # rider 25
            10,  # car 26
            11,  # truck 27
            11,  # bus 28
            15, 15,
            12,  # train 31
            13,  # motorcycle 32
            14,  # bicycle 33
        )
    )
    target = np.array(target)  # Convert target to numpy array
    target = label_map[target]  # Apply label map
    return torch.tensor(target, dtype=torch.long)

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 16  # Based on the label_map (0-15)
    
    # Configuration
    root_dir = args.data_path
    batch_size = args.batch_size
    num_epochs = args.epochs
    lr = args.lr #0.0006

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
        root=root_dir,
        split="train",
        mode="fine",
        target_type="semantic",
        transform=transform,
        target_transform=target_transform
    )

    val_dataset = Cityscapes(
        root=root_dir,
        split="val",
        mode="fine",
        target_type="semantic",
        transform=transform,
        target_transform=target_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    # Model setup
    model = create_seg_model(args.model_type, "cityscapes", pretrained=False)
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
        print(f"Resumed training from checkpoint: {args.resume}")
    model.to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3, verbose=True)
    
    # Training and Validation functions
    def train_one_epoch(model, loader, optimizer, criterion, device):
        model.train()
        total_loss = 0
        for images, targets in loader:
            images, targets = images.to(device), targets.squeeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(pixel_values=images).logits
            outputs = F.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)  # Resize to target size
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
                outputs = model(pixel_values=images).logits
                outputs = F.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)  # Resize to target size
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                preds = outputs.argmax(dim=1)
                stats = iou(preds, targets)
                interaction.update(stats["i"])
                union.update(stats["u"])
                
                acc = compute_metrics(preds.cpu().numpy(), targets.cpu().numpy(), NUM_CLASSES)
                val_accs.append(acc)
                
        return total_loss/len(loader), sum(val_accs)/len(val_accs)

    # Experiment tracking setup
    wandb.init(
        project="semantic-segmentation",
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "save_interval": args.save_interval,
            "dataset": root_dir,
        },
    )
    
    # Training loop
    best_acc = 0
    best_ckpt_path = ''
    last_ckpt_path = ''
    os.makedirs(os.path.join(args.save_dir, 'ckpt_interval'), exist_ok=True)
    
    for epoch in range(num_epochs):
        interaction = AverageMeter(is_distributed=False)
        union = AverageMeter(is_distributed=False)
        iou = SegIOU(17, ignore_index=16)
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE, interaction, union, iou)
        val_iou = (interaction.sum / union.sum).cpu().mean().item() * 100
        
        scheduler.step(val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        print(f"Validation: Loss = {val_loss:.4f}, IoU = {val_iou:.4f}, Accuracy = {val_acc:.4f}")

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_accuracy": val_acc
        })

        # Save model checkpoint
        if best_acc < val_acc:
            best_acc = val_acc
            ckpt_epoch = epoch
            if os.path.exists(best_ckpt_path):
                os.remove(best_ckpt_path)
            best_ckpt_path = os.path.join(args.save_dir, f"best_model_epoch_{epoch}_iou_{avg_val_iou}_acc_{avg_val_acc}.pth")
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"Best model saved at {best_ckpt_path}")
        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, 'ckpt_interval', f"model_epoch_{epoch}_iou_{avg_val_iou}_acc_{avg_val_acc}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model periodically saved at {checkpoint_path}")
        if os.path.exists(last_ckpt_path):
            os.remove(last_ckpt_path)
        last_ckpt_path = os.path.join(args.save_dir, f"last_model_epoch_{epoch}_iou_{avg_val_iou}_acc_{avg_val_acc}.pth")
        torch.save(model.state_dict(), last_ckpt_path)
        
        if epoch - ckpt_epoch > args.early_stop:
            print(f"Early Stopping at epoch {epoch} because of no improvement after {args.early_stop}")
            break
        
    wandb.finish()


# Main training script
if __name__ == "__main__":
    main()