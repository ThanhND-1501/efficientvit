# File: semantic_segmentation.py
# python semantic_segmentation.py --dataset_path ./dataset --output_dir ./outputs --epochs 50 --save_interval 5
# python semantic_segmentation.py --dataset_path ./dataset --output_dir ./outputs --resume_from ./outputs/latest_checkpoint.pth


import argparse
import os
from datetime import datetime
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import SegformerImageProcessor
from sklearn.metrics import jaccard_score, accuracy_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from efficientvit.seg_model_zoo import create_seg_model

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset root directory.")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for training and validation.")
    parser.add_argument('--learning_rate', type=float, default=0.0006, help="Learning rate for the optimizer.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--log_dir', type=str, default='tensorboard_logs', help="Path to save TensorBoard logs.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save model weights.")
    parser.add_argument('--resume_from', type=str, default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument('--pretrained', action='store_true', help="Use pretrained weights for the model.")
    parser.add_argument('--model_name', type=str, default="b2", help="Model name (e.g., b2, b3) for EfficientViT.")
    parser.add_argument('--dataset_name', type=str, default="cityscapes", help="Dataset name for the model.")
    parser.add_argument('--save_interval', type=int, default=5, help="Interval (in epochs) for saving checkpoints.")
    return parser.parse_args()

def compute_metrics(preds, labels, num_classes=7):
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    iou = jaccard_score(labels_flat, preds_flat, average='macro', labels=range(num_classes))
    accuracy = accuracy_score(labels_flat, preds_flat)
    return iou, accuracy

class SegmentationTransforms:
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        ])

    def __call__(self, image, mask):
        image = self.transforms(image)
        return image, mask

class SemanticSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_processor, transform=None, train=True):
        sub_path = "training" if train else "validation"
        self.img_dir = os.path.join(root_dir, "images", sub_path)
        self.ann_dir = os.path.join(root_dir, "annotations", sub_path)
        self.image_processor = image_processor
        self.transform = transform

        self.images = sorted(os.listdir(self.img_dir))
        self.annotations = sorted(os.listdir(self.ann_dir))
        assert len(self.images) == len(self.annotations), "Images and annotations must match."

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))
        if self.transform:
            image, segmentation_map = self.transform(image, segmentation_map)
        encoded_inputs = self.image_processor(image, segmentation_map, return_tensors="pt")
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()
        return encoded_inputs

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_processor = SegformerImageProcessor(reduce_labels=False, size=(1280, 800))
    transform = SegmentationTransforms()

    train_dataset = SemanticSegmentationDataset(
        root_dir=args.dataset_path,
        image_processor=image_processor,
        transform=transform,
        train=True
    )
    valid_dataset = SemanticSegmentationDataset(
        root_dir=args.dataset_path,
        image_processor=image_processor,
        transform=None,
        train=False
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)

    model = create_seg_model(args.model_name, args.dataset_name, pretrained=args.pretrained).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)

    # Resume training if checkpoint is provided
    start_epoch = 0
    if args.resume_from:
        print(f"Resuming training from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    writer = SummaryWriter(args.log_dir)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_dataloader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values)
            upsampled_outputs = nn.functional.interpolate(
                outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            loss = criterion(upsampled_outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)

        scheduler.step(avg_epoch_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_ious, val_accuracies = [], []
        with torch.no_grad():
            for batch in valid_dataloader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(pixel_values)
                upsampled_outputs = nn.functional.interpolate(
                    outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                loss = criterion(upsampled_outputs, labels)
                val_loss += loss.item()

                predicted = upsampled_outputs.argmax(dim=1)
                batch_iou, batch_accuracy = compute_metrics(predicted.cpu().numpy(), labels.cpu().numpy())
                val_ious.append(batch_iou)
                val_accuracies.append(batch_accuracy)

        avg_val_loss = val_loss / len(valid_dataloader)
        avg_val_iou = sum(val_ious) / len(val_ious)
        avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)

        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('IoU/val', avg_val_iou, epoch)
        writer.add_scalar('Accuracy/val', avg_val_accuracy, epoch)

        # Save checkpoint at specific intervals
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, 'latest_checkpoint.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
