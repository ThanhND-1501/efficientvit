import argparse
import math
import os
import sys

import cv2
import numpy as np
import torch
from cityscapes_pt import CityscapesDataset, Resize, ToTensor, get_canvas
from PIL import Image
from time import time
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.models.utils import resize
from efficientvit.seg_model_zoo import create_seg_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="assets/video/sample_video.mp4")
    parser.add_argument("--dataset", type=str, default="cityscapes", choices=["cityscapes"])
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--model", type=str, default="b0")
    parser.add_argument("--weight_url", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=".demo/efficientvit_seg_demo.mp4")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    LOGS = "efficientvit.log"

    cap = cv2.VideoCapture(args.video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

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

    model = create_seg_model(args.model, weight_url=args.weight_url).cuda()
    model.eval()

    total_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        data = transform({"image": image, "label": np.ones_like(image)})["image"]

        with torch.inference_mode():
            data = torch.unsqueeze(data, dim=0).cuda()
            start = time()
            output = model(data)
            end = time() - start
            total_time += end
            if output.shape[-2:] != image.shape[:2]:
                output = resize(output, size=image.shape[:2])
            output = torch.argmax(output, dim=1).cpu().numpy()[0]
            canvas = get_canvas(image, output, class_colors)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            out.write(canvas)
    with open(LOGS, "a") as f:
        f.write(f"\nRunning time: {total_time} (s)")

    cap.release()
    out.release()

if __name__ == "__main__":
    main()