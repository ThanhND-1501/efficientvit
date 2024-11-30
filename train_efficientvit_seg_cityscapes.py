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
import torchvision.transforms.functional as F
from tqdm import tqdm
from sklearn.metrics import jaccard_score, accuracy_score
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
from typing import Any, Optional
from sys import argv

# Custom functions
def compute_metrics(preds, labels, num_classes=20):
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    iou = jaccard_score(labels_flat, preds_flat, average='macro', labels=range(num_classes))
    accuracy = accuracy_score(labels_flat, preds_flat)
    return iou, accuracy

class Resize(object):
    def __init__(
        self,
        crop_size: Optional[tuple[int, int]],
        interpolation: Optional[int] = cv2.INTER_CUBIC,
    ):
        self.crop_size = crop_size
        self.interpolation = interpolation

    def __call__(self, feed_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.crop_size is None or self.interpolation is None:
            return feed_dict

        image, target = feed_dict["image"], feed_dict["label"]
        height, width = self.crop_size

        h, w, _ = image.shape
        if width != w or height != h:
            image = cv2.resize(
                image,
                dsize=(width, height),
                interpolation=self.interpolation,
            )
        return {
            "image": image,
            "label": target,
        }


class ToTensor(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, feed_dict: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        image, mask = feed_dict["image"], feed_dict["label"]
        image = image.transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)
        image = torch.as_tensor(image, dtype=torch.float32).div(255.0)
        mask = torch.as_tensor(mask, dtype=torch.int64)
        image = F.normalize(image, self.mean, self.std, self.inplace)
        return {
            "image": image,
            "label": mask,
        }

# Dataset class
class CityscapesDataset(Dataset):
    """
    Cityscapes semantic segmentation dataset.
    Images and labels are organized as per the official Cityscapes directory structure.
    """

    label_map = np.array(
        (
            19,
            19,
            19,
            19,
            19,
            19,
            19,
            0,  # road 7
            1,  # sidewalk 8
            19,
            19,
            2,  # building 11
            3,  # wall 12
            4,  # fence 13
            19,
            19,
            19,
            5,  # pole 17
            19,
            6,  # traffic light 19
            7,  # traffic sign 20
            8,  # vegetation 21
            9,  # terrain 22
            10,  # sky 23
            11,  # person 24
            12,  # rider 25
            13,  # car 26
            14,  # truck 27
            15,  # bus 28
            19,
            19,
            16,  # train 31
            17,  # motorcycle 32
            18,  # bicycle 33
        )
    )

    
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Root directory of the Cityscapes dataset.
            split (str): Dataset split - 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # File paths for images and labels
        self.img_dir = os.path.join(root_dir, "leftImg8bit", split)
        self.ann_dir = os.path.join(root_dir, "gtFine", split)

        # Gather all image paths
        self.images = []
        self.labels = []
        self.samples = []
        for city in os.listdir(self.img_dir):
            city_img_dir = os.path.join(self.img_dir, city)
            city_ann_dir = os.path.join(self.ann_dir, city)
            for file_name in os.listdir(city_img_dir):
                if file_name.endswith("_leftImg8bit.png"):
                    img_path = os.path.join(city_img_dir, file_name)
                    label_path = os.path.join(city_ann_dir, file_name.replace("_leftImg8bit.png", "_gtFine_labelIds.png"))
                    self.images.append(img_path)
                    self.labels.append(label_path)
                    self.samples.append((img_path, label_path))

        assert len(self.images) == len(self.labels), "Mismatch between images and labels!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and label
        image = np.array(Image.open(self.images[idx]).convert("RGB"))
        label = np.array(Image.open(self.labels[idx]))
        label = self.label_map[label]

        # Apply transformations
        if self.transform:
            image, label = self.transform(image, label)

        return {"image": image, "label": label}


# Transformations
class CityscapesTransforms:
    def __call__(self, image, label):
        # Resize to a fixed size for uniform input
        # resize = transforms.Resize((512, 1024))  # Adjust size as per model's input requirement
        transform_size_tensor = transforms.Compose(
            [
                Resize((512, 1024)),
                ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        feed_dict = transform_size_tensor({'image':image, 'label':label})
        image, label = feed_dict['image'], feed_dict['label']

        # Apply random horizontal flip
        if torch.rand(1).item() > 0.5:
            image = transforms.functional.hflip(image)
            label = transforms.functional.hflip(label)

        # Convert image to tensor and normalize
        # image = transforms.ToTensor()(image)
        # image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

        # Convert label to NumPy and then to tensor
        label = torch.tensor(np.array(label), dtype=torch.long)

        return image, label

# Main training script
if __name__ == "__main__":
    # Configuration
    root_dir = argv[1]
    print('ROOT_DIR:', root_dir)
    log_dir = "tensorboard_log_dir"
    batch_size = 8
    num_epochs = argv[2]
    num_classes = 20  # Cityscapes has 19 classes
    lr = 0.0006
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loaders
    transforms_cityscapes = CityscapesTransforms()

    train_dataset = CityscapesDataset(root_dir=root_dir, split="train", transform=transforms_cityscapes)
    val_dataset = CityscapesDataset(root_dir=root_dir, split="val", transform=transforms_cityscapes)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)


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
