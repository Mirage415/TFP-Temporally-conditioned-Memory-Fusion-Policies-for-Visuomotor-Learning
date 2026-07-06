import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_a1_example() -> dict:
    """Creates a random input example for the A1 single-arm policy."""
    return {
        "cam_0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "cam_1": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "state": np.random.rand(7),
        "prompt": "swap the position of the banana and the white board eraser",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        if image.size > 0 and float(np.nanmax(image)) <= 1.0 + 1e-6:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got {image.shape}.")
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image


@dataclasses.dataclass(frozen=True)
class A1Inputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["cam_0"])
        wrist_image = _parse_image(data["cam_1"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class A1Outputs(transforms.DataTransformFn):
    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
