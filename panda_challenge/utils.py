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


def tile(img, sz=128, N=16, mask=None):
    shape = img.shape
    pad0, pad1 = (sz - shape[0] % sz) % sz, (sz - shape[1] % sz) % sz
    img = np.pad(
        img,
        [[pad0//2, pad0-pad0//2], [pad1//2, pad1-pad1//2], [0, 0]],
        constant_values=255)
    if mask is not None:
        mask = np.pad(
            mask,
            [[pad0//2, pad0-pad0//2], [pad1//2, pad1-pad1//2], [0, 0]],
            constant_values=0)
    img = img.reshape(img.shape[0]//sz, sz, img.shape[1]//sz, sz, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)
    if mask is not None:
        mask = mask.reshape(mask.shape[0]//sz, sz, mask.shape[1]//sz, sz, 3)
        mask = mask.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)
    if len(img) < N:
        if mask is not None:
            mask = np.pad(
                mask,
                [[0, N-len(img)], [0, 0], [0, 0], [0, 0]],
                constant_values=0)
        img = np.pad(
            img,
            [[0, N-len(img)], [0, 0], [0, 0], [0, 0]],
            constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:N]
    img = img[idxs]
    data = {'img': img}
    if mask is not None:
        mask = mask[idxs]
        mask = [m[..., 0] for m in mask]
        data['mask'] = mask
    return data
