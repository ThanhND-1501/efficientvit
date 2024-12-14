# !python train_mobilevit_seg_cityscapes.py --data_path --epochs --batch_size --save_dir --save_interval --resume --lr
import os
import argparse
import wandb

from dataset import *
from utils import *
from efficientvit.apps.utils import AverageMeter

from torch.utils.data import DataLoader
from torch import nn
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from transformers import MobileViTForImageClassification

# Define command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--model_type", type=str, default="b1", required=True, help="Type of EfficientViT model.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--save_interval", type=int, default=5, help="Interval (in epochs) to save checkpoints.")
    parser.add_argument("--resume", type=str, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--lr", type=float, default=0.0006, help="Learning rate for the optimizer.")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuration
    root_dir = args.data_path
    print('ROOT_DIR:', root_dir)
    log_dir = "tensorboard_log_dir"
    batch_size = args.batch_size
    num_epochs = args.epochs
    num_classes = 20  # Cityscapes has 19 classes
    lr = args.lr #0.0006

    # Data loaders
    transforms_cityscapes = CityscapesTransforms()

    train_dataset = CityscapesDataset(root_dir=root_dir, split="train", transform=transforms_cityscapes)
    val_dataset = CityscapesDataset(root_dir=root_dir, split="val", transform=transforms_cityscapes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    # Model setup
    model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
        print(f"Resumed training from checkpoint: {args.resume}")
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3, verbose=True)

    # TensorBoard setup
    writer = SummaryWriter(log_dir)
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
    
    # Training loop
    best_acc = 0
    best_ckpt_path = ''
    last_ckpt_path = ''
    os.makedirs(os.path.join(args.save_dir, 'ckpt_interval'), exist_ok=True)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            outputs = nn.functional.interpolate(outputs, size=labels.shape[1:], mode="bilinear", align_corners=False)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_loss, epoch)
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_accs = []
        interaction = AverageMeter(is_distributed=False)
        union = AverageMeter(is_distributed=False)
        iou = SegIOU(17, ignore_index=16)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images)
                outputs = nn.functional.interpolate(outputs, size=labels.shape[1:], mode="bilinear", align_corners=False)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                stats = iou(preds, labels)
                interaction.update(stats["i"])
                union.update(stats["u"])
                
                acc = compute_metrics(preds.cpu().numpy(), labels.cpu().numpy(), num_classes)
                val_accs.append(acc)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = (interaction.sum / union.sum).cpu().mean().item() * 100
        avg_val_acc = sum(val_accs) / len(val_accs)
        
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("IoU/Val", avg_val_iou, epoch)
        writer.add_scalar("Accuracy/Val", avg_val_acc, epoch)

        print(f"Validation: Loss = {avg_val_loss:.4f}, IoU = {avg_val_iou:.4f}, Accuracy = {avg_val_acc:.4f}")

        wandb.log({
            "train_loss": avg_loss,
            "val_loss": avg_val_loss,
            "val_iou": avg_val_iou,
            "val_accuracy": avg_val_acc
        })
        
        # Adjust learning rate
        scheduler.step(avg_val_loss)

        # Save model checkpoint
        if best_acc < avg_val_acc:
            best_acc = avg_val_acc
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
    wandb.finish()

# Main training script
if __name__ == "__main__":
    main()