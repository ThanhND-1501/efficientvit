import os
import torch
import cv2
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Optional, Union, Tuple
import torchvision.transforms.functional as F

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

# Transformations
class CityscapesTransforms:
    def __init__(self, size: Union[Tuple, int]=(512, 1024)):
        self.size = size
    
    def __call__(self, image, label):
        # Resize to a fixed size for uniform input
        # resize = transforms.Resize((512, 1024))  # Adjust size as per model's input requirement
        transform_size_tensor = transforms.Compose(
            [
                transforms.,
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

# Dataset class
class CityscapesDataset(Dataset):
    """
    Cityscapes semantic segmentation dataset.
    Images and labels are organized as per the official Cityscapes directory structure.
    """

    label_map = np.array(
        (
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            0,  # road 7
            1,  # sidewalk 8
            16,
            16,
            2,  # building 11
            2,  # wall 12
            2,  # fence 13
            16,
            16,
            16,
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
            16,
            16,
            13,  # train 31
            14,  # motorcycle 32
            15,  # bicycle 33
        )
    )
    
    class_color = {'road': (128, 64, 128),
               'sidewalk': (244, 35, 232),
               'building': (70, 70, 70),
               'wall': (102, 102, 156),
               'fence': (190, 153, 153),
               'pole': (153, 153, 153),
               'traffic light': (250, 170, 30),
               'traffic sign': (220, 220, 0),
               'vegetation': (107, 142, 35),
               'terrain': (152, 251, 152),
               'sky': (70, 130, 180),
               'person': (220, 20, 60),
               'rider': (255, 0, 0),
               'car': (0, 0, 142),
               'truck': (0, 0, 70),
               'bus': (0, 60, 100),
               'train': (0, 80, 100),
               'motorcycle': (0, 0, 230),
               'bicycle': (119, 11, 32)}

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

def get_canvas(
    image: np.ndarray,
    mask: np.ndarray,
    colors: tuple | list,
    opacity=0.5,
) -> np.ndarray:
    image_shape = image.shape[:2]
    mask_shape = mask.shape
    if image_shape != mask_shape:
        mask = cv2.resize(mask, dsize=(image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    seg_mask = np.zeros_like(image, dtype=np.uint8)
    for k, color in enumerate(colors):
        seg_mask[mask == k, :] = color
    canvas = seg_mask * opacity + image * (1 - opacity)
    canvas = np.asarray(canvas, dtype=np.uint8)
    return canvas