import numpy as np
import os
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
import albumentations as A
from torch.utils.data import Dataset
import torch
import openslide
from .utils import crop_around_mask
import json


class ClassifcationDatasetSimpleTrain(Dataset):

    CROP_HIGHEST_ZOOM = 32

    def __init__(
        self,
        data_df,
        transforms_json,
        image_dir,
        mask_dir,
            crop_size=512):
        """Prepares pytorch dataset for training
        Crops around mask and returns ISUP score for the slide

        Args:
            data_df (pd.DataFrame): data.frame with slides id and labels.
            augmentations (albumentations.compose): augmentations.
            image_dir (str): folder with images.
            mask_dir (str): folder with masks.
            crop_size(int): crop size around mask. Default: 512
        Returns
            Dataset

        """
        self.data_df = data_df
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.crop_size = crop_size
        self.transforms = self._get_aug(transforms_json)

    def _get_aug(self, arg):
        with open(arg) as f:
            return A.from_dict(json.load(f))

    def __getitem__(self, idx):
        """Will load the mask, get random coordinates around/with the mask,
        load the image by coordinates
        """
        slide_id = self.data_df.image_id.values[idx]
        isup_grade = self.data_df.isup_grade.values[idx]
        mask = openslide.OpenSlide(os.path.join(
            self.mask_dir,
            f'{slide_id}_mask.tiff'))
        image = openslide.OpenSlide(os.path.join(
            self.image_dir,
            f'{slide_id}.tiff'))
        k = int(mask.level_downsamples[-1])
        mask_full = mask.read_region(
            (0, 0),
            mask.level_count - 1,
            mask.level_dimensions[-1])
        mask_full = np.asarray(mask_full).astype(bool)[..., 0]
        crop_coords = crop_around_mask(
            mask_full,
            ClassifcationDatasetSimpleTrain.CROP_HIGHEST_ZOOM,
            ClassifcationDatasetSimpleTrain.CROP_HIGHEST_ZOOM)
        image_slice = image.read_region(
            (k*crop_coords['x_min'], k*crop_coords['y_min']),
            0,
            (self.crop_size, self.crop_size))
        image_slice = np.asarray(image_slice)[..., :3]
        augmented = self.transforms(image=image_slice)
        image = augmented['image']
        data = {'features': tensor_from_rgb_image(image).float(),
                'targets': torch.tensor(isup_grade)}
        return(data)
