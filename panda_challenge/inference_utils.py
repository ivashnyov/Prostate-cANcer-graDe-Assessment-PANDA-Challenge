import os
import torch
import openslide
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from pytorch_toolbelt.utils.torch_utils import to_numpy
from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
import cv2


def prepare_single_slice(image, expand_dim=False, as_tensor=False):
    image = A.Normalize()(image=image)['image']
    image = image.transpose((2, 0, 1))
    if expand_dim:
        image = np.expand_dims(image, 0)
    if as_tensor:
        image = torch.from_numpy(image)
    return(image)


def normalize_single_slice(image):
    image = A.Normalize()(image=image)['image']
    return(image)


def crop_white(image: np.ndarray) -> np.ndarray:
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    ys, = (image.min((1, 2)) != 255).nonzero()
    xs, = (image.min(0).min(1) != 255).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return [0, image.shape[0], 0, image.shape[1]]
    return [ys.min(), ys.max() + 1, xs.min(), xs.max() + 1]


class InferenceSingleImage(Dataset):
    def __init__(
        self,
        crops,
        slide_id,
        image_dir,
        crop_size=512,
            bg_thr=0.25):
        self.crops = crops
        self.slide_id = slide_id
        self.image_dir = image_dir
        self.crop_size = crop_size
        self.bg_thr = bg_thr

    def __len__(self):
        return(len(self.crops))

    def __repr__(self):
        return(f'InferenceSingleImage({self.slide_id})')

    def __getitem__(self, idx):
        image = openslide.OpenSlide(os.path.join(
            self.image_dir,
            f'{self.slide_id}.tiff'))
        crop_x = self.crops[idx][0]
        crop_y = self.crops[idx][1]
        image_slice = image.read_region(
            (crop_x, crop_y),
            0,
            (self.crop_size, self.crop_size))
        image_slice = np.asarray(image_slice)[..., :3]
        white_bg = (image_slice == 255).sum() / (3*(self.crop_size)**2)
        if white_bg > self.bg_thr:
            need_process = False
        else:
            need_process = True
        if need_process:
            image_slice = prepare_single_slice(
                image_slice,
                as_tensor=True).float()
        else:
            image_slice = torch.from_numpy(
                image_slice.transpose((2, 0, 1))).float()
        data = {'image': image_slice,
                'coords': self.crops[idx],
                'need_to_process': need_process}
        return(data)


def get_slice_heatmap(
    slide_id,
    image_dir,
    model,
    crop_size=512,
    tile_step=512,
    bsize=16,
    num_workers=4,
    normalize=True,
    *args,
        **kwargs):
    image = openslide.OpenSlide(os.path.join(
        image_dir,
        f'{slide_id}.tiff'))
    x, y = image.level_dimensions[0][::-1]
    tiler = ImageSlicer(
        (x, y),
        tile_size=(crop_size, crop_size),
        tile_step=(tile_step, tile_step),
        weight='mean')
    merger = CpuTileMerger(
        tiler.target_shape,
        1,
        tiler.weight,
        normalize=normalize)
    dataset = InferenceSingleImage(
        tiler.crops,
        slide_id,
        image_dir,
        crop_size=crop_size,
        **kwargs)
    dataloader = DataLoader(
        dataset,
        batch_size=bsize,
        pin_memory=True,
        num_workers=num_workers)
    pseudo_mask = np.ones((1, crop_size, crop_size))
    with torch.no_grad():
        for data_b in tqdm(dataloader, total=len(dataloader)):
            need_processing = data_b['need_to_process']
            if need_processing.sum() > 0:
                pred_batch = model(data_b['image'][need_processing].cuda())
                pred_batch = torch.nn.Softmax(dim=1)(pred_batch).cpu().numpy()
                pred_batch = np.stack([pseudo_mask*np.argmax(pred_batch[idx])
                                       for idx in range(pred_batch.shape[0])])
                merger.integrate_batch(
                    pred_batch,
                    data_b['coords'][need_processing].cpu().numpy())
    merged_mask = np.moveaxis(to_numpy(merger.merge()), 0, -1).astype(np.uint8)
    merged_mask = tiler.crop_to_orignal_size(merged_mask)
    merged_mask = cv2.resize(
        merged_mask,
        image.level_dimensions[-1],
        interpolation=cv2.INTER_AREA)
    return(merged_mask)


class CpuTileMerger:
    """
    Helper class to merge final image on CPU
    """

    def __init__(self, image_shape, channels, weight, normalize=True):
        """
        :param image_shape: Shape of the source image
        :param image_margin:
        :param weight: Weighting matrix
        """
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.normalize = normalize
        self.weight = np.expand_dims(weight, axis=0)
        self.channels = channels
        self.image = np.zeros(
            (channels, self.image_height, self.image_width))
        self.norm_mask = np.zeros((1, self.image_height, self.image_width))

    def integrate_batch(self, batch: torch.Tensor, crop_coords):
        """
        Accumulates batch of tile predictions
        :param batch: Predicted tiles
        :param crop_coords: Corresponding tile crops w.r.t to original image
        """
        if len(batch) != len(crop_coords):
            raise ValueError("Number of images in batch does not correspond to number of coordinates")

        for tile, (x, y, tile_width, tile_height) in zip(batch, crop_coords):
            self.image[:, y:y + tile_height, x:x + tile_width] += tile * self.weight
            self.norm_mask[:, y:y + tile_height, x:x + tile_width] += self.weight

    def merge(self) -> torch.Tensor:
        if self.normalize:
            return self.image / self.norm_mask
        else:
            return self.image
