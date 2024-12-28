import argparse
import math
import os
import sys

import cv2
import numpy as np
import torch
from cityscapes_pt import CityscapesDataset, Resize, ToTensor, get_canvas
from PIL import Image
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.models.utils import resize
from efficientvit.seg_model_zoo import create_seg_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="assets/fig/indoor.jpg")
    parser.add_argument("--dataset", type=str, default="cityscapes", choices=["cityscapes"])
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--model", type=str, default="b0")
    parser.add_argument("--weight_url", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=".demo/efficientvit_seg_demo.png")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    image = np.array(Image.open(args.image_path).convert("RGB"))
    data = image
    if args.dataset == "cityscapes":
        transform = transforms.Compose(
            [
                Resize((args.crop_size, args.crop_size * 2)),
                ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        class_colors = CityscapesDataset.class_color.values()
    else:
        raise NotImplementedError
    data = transform({"image": data, "label": np.ones_like(data)})["image"]

    model = create_seg_model(args.model, weight_url=args.weight_url).cuda()
    model.eval()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with torch.inference_mode():
        data = torch.unsqueeze(data, dim=0).cuda()
        output = model(data)
        # resize the output to match the shape of the mask
        if output.shape[-2:] != image.shape[:2]:
            output = resize(output, size=image.shape[:2])
        output = torch.argmax(output, dim=1).cpu().numpy()[0]
        canvas = get_canvas(image, output, class_colors)
        canvas = Image.fromarray(canvas).save(args.output_path)


if __name__ == "__main__":
    main()
