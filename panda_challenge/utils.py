import random
import numpy as np
import torch.nn as nn
from pdb import set_trace as st
from typing import List
import torch


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


def tileV2(img, mode=0, tile_size=256, n_tiles=36):
    '''
    Getting tiles from the image
    source: https://www.kaggle.com/haqishen/train-efficientnet-b0-w-36-tiles-256-lb0-87
    '''
    result = []
    h, w, c = img.shape
    pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
    pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

    img2 = np.pad(
        img,
        [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w//2], [0, 0]],
        constant_values=255
        )
    img3 = img2.reshape(
        img2.shape[0] // tile_size,
        tile_size,
        img2.shape[1] // tile_size,
        tile_size,
        3
    )

    img3 = img3.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, 3)
    n_tiles_with_info = (img3.reshape(img3.shape[0], -1).sum(1) < tile_size ** 2 * 3 * 255).sum()  # Full white image
    if len(img3) < n_tiles:
        img3 = np.pad(img3, [[0, n_tiles-len(img3)], [0, 0], [0, 0], [0, 0]], constant_values=255)
    idxs = np.argsort(img3.reshape(img3.shape[0], -1).sum(-1))[:n_tiles]
    img3 = img3[idxs]
    for i in range(len(img3)):
        result.append({'img': img3[i], 'idx': i})
    return result, n_tiles_with_info >= n_tiles


BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def freeze_to(module: nn.Module, n: int, freeze_bn: bool = False) -> None:
    # source: https://github.com/belskikh/kekas/
    layers = list(module.children())
    for l in layers[:n]:
        for module in flatten_layer(l):
            if freeze_bn or not isinstance(module, BN_TYPES):
                set_grad(module, requires_grad=False)

    for l in layers[n:]:
        for module in flatten_layer(l):
            set_grad(module, requires_grad=True)


def freeze(module: nn.Module, freeze_bn: bool = False, n: int = -1) -> None:
    # source: https://github.com/belskikh/kekas/
    freeze_to(module=module, n=n, freeze_bn=freeze_bn)


def unfreeze(module: nn.Module) -> None:
    # source: https://github.com/belskikh/kekas/
    layers = list(module.children())
    for l in layers:
        for module in flatten_layer(l):
            set_grad(module, requires_grad=True)


def set_grad(module: nn.Module, requires_grad: bool) -> None:
    # source: https://github.com/belskikh/kekas/
    for param in module.parameters():
        param.requires_grad = requires_grad


def flatten_layer(layer: nn.Module) -> List[nn.Module]:
    # source: https://github.com/belskikh/kekas/
    if len(list(layer.children())):
        layers = []
        for children in children_and_parameters(layer):
            layers += flatten_layer(children)
        return layers
    else:
        return [layer]


def to_numpy(data: torch.Tensor) -> np.ndarray:
    # source: https://github.com/belskikh/kekas/
    return data.detach().cpu().numpy()


class ParameterModule(nn.Module):
    # source: https://github.com/belskikh/kekas/
    """Register a lone parameter `p` in a module."""

    def __init__(self, p: nn.Parameter):
        super().__init__()
        self.val = p

    def forward(self, x):
        return x


# https://github.com/fastai/fastai/blob/6778fd518e95ea8e1ce1e31a2f96590ee254542c/fastai/torch_core.py#L149
def children_and_parameters(m: nn.Module):
    """Return the children of `m` and its direct parameters not registered in modules."""
    children = list(m.children())
    children_p = sum([[id(p) for p in c.parameters()] for c in m.children()], [])
    for p in m.parameters():
        if id(p) not in children_p:
            st()
            children.append(ParameterModule(p))
    return children
