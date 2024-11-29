# File: semantic_segmentation.py
# python semantic_segmentation.py --dataset_path ./dataset --output_dir ./outputs --epochs 50 --save_interval 5
# python semantic_segmentation.py --dataset_path ./dataset --output_dir ./outputs --resume_from ./outputs/latest_checkpoint.pth
# python semantic_segmentation.py --dataset_path /path/to/Cityscapes --output_dir ./outputs --epochs 50 --batch_size 4
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from efficientvit.seg_model_zoo import create_seg_model
from torch import nn
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import jaccard_score, accuracy_score
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Custom functions
def compute_metrics(preds, labels, num_classes=19):
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    iou = jaccard_score(labels_flat, preds_flat, average='macro', labels=range(num_classes))
    accuracy = accuracy_score(labels_flat, preds_flat)
    return iou, accuracy

# Dataset class
class SemanticSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        sub_path = "train" if train else "val"
        self.img_dir = os.path.join(root_dir, "leftImg8bit", sub_path)
        self.ann_dir = os.path.join(root_dir, "gtFine", sub_path)
        self.transform = transform

        self.images = sorted(os.listdir(self.img_dir))
        self.annotations = sorted(os.listdir(self.ann_dir))
        assert len(self.images) == len(self.annotations), "Images and annotations mismatch!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        label = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))

        if self.transform:
            image, label = self.transform(image, label)

        return {"image": image, "label": torch.tensor(label, dtype=torch.long)}

# Transformations
class SegmentationTransforms:
    def __call__(self, image, label):
        # Apply random horizontal flip
        if torch.rand(1).item() > 0.5:
            image = transforms.functional.hflip(image)
            label = transforms.functional.hflip(label)

        # Convert to tensor and normalize
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

# Main training script
if __name__ == "__main__":
    # Configuration
    root_dir = "/kaggle/input/cityscapes-dataset"
    log_dir = "tensorboard_log_dir"
    batch_size = 4
    num_epochs = 50
    num_classes = 19  # Cityscapes has 19 classes
    lr = 0.0006
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loaders
    train_transforms = SegmentationTransforms()
    train_dataset = SemanticSegmentationDataset(root_dir, transform=train_transforms, train=True)
    val_dataset = SemanticSegmentationDataset(root_dir, transform=None, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model setup
    model = create_seg_model("b2", "cityscapes", pretrained=False)
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3, verbose=True)

    # TensorBoard setup
    writer = SummaryWriter(log_dir)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
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
        val_ious, val_accs = [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images)
                outputs = nn.functional.interpolate(outputs, size=labels.shape[1:], mode="bilinear", align_corners=False)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                iou, acc = compute_metrics(preds.cpu().numpy(), labels.cpu().numpy(), num_classes)
                val_ious.append(iou)
                val_accs.append(acc)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = sum(val_ious) / len(val_ious)
        avg_val_acc = sum(val_accs) / len(val_accs)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("IoU/Val", avg_val_iou, epoch)
        writer.add_scalar("Accuracy/Val", avg_val_acc, epoch)
        print(f"Validation: Loss = {avg_val_loss:.4f}, IoU = {avg_val_iou:.4f}, Accuracy = {avg_val_acc:.4f}")

        # Adjust learning rate
        scheduler.step(avg_val_loss)

        # Save model checkpoint
        checkpoint_path = f"model_epoch_{epoch + 1}_loss_{avg_val_loss:.4f}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved at {checkpoint_path}")
