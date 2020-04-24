import random
import numpy as np


def crop_around_mask(mask, height, width):
    """Return coordinates of non-empty crop
    on the mask

    Args:
        mask (np.ndarray)
        height (int): target height
        width (int): target width
    """
    mask_height, mask_width = mask.shape[:2]
    if height > mask_height or width > mask_width:
        raise ValueError(
            "Crop size ({},{}) is larger than image ({},{})".format(
                height, width, mask_height, mask_width
            )
        )
    if mask.sum() == 0:
        x_min = random.randint(0, mask_width - width)
        y_min = random.randint(0, mask_height - height)
    else:
        mask = mask.sum(axis=-1) if mask.ndim == 3 else mask
        non_zero_yx = np.argwhere(mask)
        y, x = random.choice(non_zero_yx)
        x_min = x - random.randint(0, width - 1)
        y_min = y - random.randint(0, height - 1)
        x_min = np.clip(x_min, 0, mask_width - width)
        y_min = np.clip(y_min, 0, mask_height - height)
    x_max = x_min + width
    y_max = y_min + height
    return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
