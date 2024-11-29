from functools import partial
from typing import Callable, Optional

from efficientvit.models.efficientvit import (
    EfficientViTSeg,
    efficientvit_seg_b0,
    efficientvit_seg_b1,
    efficientvit_seg_b2,
    efficientvit_seg_b3,
    efficientvit_seg_l1,
    efficientvit_seg_l2,
)
from efficientvit.models.nn.norm import set_norm_eps
from efficientvit.models.utils import load_state_dict_from_file

__all__ = ["create_seg_model"]

REGISTERED_SEG_MODELS: dict[str, dict[str, Callable]] = {
    "cityscapes": {
        "b0": partial(efficientvit_seg_b0, dataset="cityscapes"),
        "b1": partial(efficientvit_seg_b1, dataset="cityscapes"),
        "b2": partial(efficientvit_seg_b2, dataset="cityscapes"),
        "b3": partial(efficientvit_seg_b3, dataset="cityscapes"),
        "l1": partial(efficientvit_seg_l1, dataset="cityscapes"),
        "l2": partial(efficientvit_seg_l2, dataset="cityscapes"),
    },
    "ade20k": {
        "b1": partial(efficientvit_seg_b1, dataset="ade20k"),
        "b2": partial(efficientvit_seg_b2, dataset="ade20k"),
        "b3": partial(efficientvit_seg_b3, dataset="ade20k"),
        "l1": partial(efficientvit_seg_l1, dataset="ade20k"),
        "l2": partial(efficientvit_seg_l2, dataset="ade20k"),
    },
}

def create_seg_model(
    model_name: str,
    dataset_name: str = "cityscapes",
    pretrained: bool = True,
    weight_url: Optional[str] = None,
    **kwargs,
) -> EfficientViTSeg:
    """
    Create a segmentation model.

    Args:
        model_name (str): The specific model variant (e.g., "b0", "b1").
        dataset_name (str): The dataset name (e.g., "cityscapes", "ade20k").
        pretrained (bool): Whether to load pretrained weights.
        weight_url (Optional[str]): Custom URL/path for the pretrained weights.
        **kwargs: Additional arguments for the model creation.

    Returns:
        EfficientViTSeg: The segmentation model.
    """
    if dataset_name not in REGISTERED_SEG_MODELS:
        raise ValueError(
            f"Dataset '{dataset_name}' is not supported. Available datasets: {list(REGISTERED_SEG_MODELS.keys())}"
        )
    if model_name not in REGISTERED_SEG_MODELS[dataset_name]:
        raise ValueError(
            f"Model '{model_name}' is not registered for dataset '{dataset_name}'. "
            f"Available models: {list(REGISTERED_SEG_MODELS[dataset_name].keys())}"
        )

    # Retrieve model constructor and checkpoint
    model_constructor = REGISTERED_SEG_MODELS[dataset_name][model_name]

    # Create the model
    model = model_constructor(**kwargs)
    set_norm_eps(model, 1e-5 if "b" in model_name else 1e-7)

    # Load pretrained weights if required
    if pretrained:
        weight_path = weight_url
        if weight_path is None:
            raise ValueError(
                f"No pretrained weights found for model '{model_name}' on dataset '{dataset_name}'."
            )
        print(f"Loading pretrained weights from {weight_path}")
        weight = load_state_dict_from_file(weight_path)
        model.load_state_dict(weight)

    return model
