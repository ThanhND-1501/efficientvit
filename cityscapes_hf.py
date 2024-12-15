from datasets import DatasetInfo, GeneratorBasedBuilder, Split, Features, Value, Array2D, Image
import numpy as np
import os
import cv2
from PIL import Image as PILImage


class HuggingfaceCityscapes(GeneratorBasedBuilder):
    """
    Hugging Face dataset implementation of Cityscapes.
    """
    
    # Dataset metadata
    BUILDER_CONFIGS = [
        {
            "name": "default",
            "description": "Cityscapes semantic segmentation dataset",
            "version": "1.0.0",
        }
    ]
    
    def _info(self):
        return DatasetInfo(
            description="Cityscapes semantic segmentation dataset with images and pixel-wise labels.",
            features=Features(
                {
                    "image": Image(),
                    "label": Array2D(shape=(None, None), dtype="int64"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://www.cityscapes-dataset.com/",
            citation="""@article{cityscapes,
                        title={The Cityscapes Dataset},
                        year={2016},
                        journal={CVPR},
                        author={Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler, Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},
                    }"""
        )

    def _split_generators(self, dl_manager):
        root_dir = self.config.data_dir  # Path to Cityscapes dataset root directory
        
        # Define splits
        return [
            Split(name=Split.TRAIN, gen_kwargs={"root_dir": root_dir, "split": "train"}),
            Split(name=Split.VALIDATION, gen_kwargs={"root_dir": root_dir, "split": "val"}),
            Split(name=Split.TEST, gen_kwargs={"root_dir": root_dir, "split": "test"}),
        ]

    def _generate_examples(self, root_dir, split):
        img_dir = os.path.join(root_dir, "leftImg8bit", split)
        ann_dir = os.path.join(root_dir, "gtFine", split)

        # Define label map
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

        # Iterate over all cities and files
        idx = 0
        for city in os.listdir(img_dir):
            city_img_dir = os.path.join(img_dir, city)
            city_ann_dir = os.path.join(ann_dir, city)

            for file_name in os.listdir(city_img_dir):
                if file_name.endswith("_leftImg8bit.png"):
                    img_path = os.path.join(city_img_dir, file_name)
                    label_path = os.path.join(city_ann_dir, file_name.replace("_leftImg8bit.png", "_gtFine_labelIds.png"))

                    # Load and process data
                    img_cv = cv2.imread(img_path)
                    image = PILImage.fromarray(img_cv)
                    label = np.array(PILImage.open(label_path))
                    label = label_map[label]  # Map labels

                    yield idx, {"image": image, "label": label}
                    idx += 1
