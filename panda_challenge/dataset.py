import numpy as np
import os
import albumentations as A
from torch.utils.data import Dataset
import torch
import openslide
from .utils import tile, tileV2
import json
import pandas as pd
import gc
import pickle


class ClassifcationDatasetMultiCrop(Dataset):
    def __init__(
        self,
        data_df,
        transforms_json,
        image_dir,
        mask_dir,
        mean=np.array([0.90949707, 0.8188697, 0.87795304]),
        std=np.array([0.36357649, 0.49984502, 0.40477625]),
        N=16,
        zoom_level=2,
        crop_size=128,
        label_smoothing=0.15,
        output_type='classification',
        use_saved=False,
        effective_N=None,
        resize_N=None,
        custom_normalization=False,
        *args,
            **kwargs):
        """Prepares pytorch dataset for training
        Generates tiles from coarse slide and returns it

        Args:
            data_df (pd.DataFrame): data.frame with slides id and labels.
            augmentations (albumentations.compose): augmentations.
            image_dir (str): folder with images.
            mask_dir (str): folder with masks.
            crop_size(int): crop size around mask. Default: 128
        Returns
            Dataset

        """
        self.data_df = pd.read_csv(data_df)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.crop_size = crop_size
        self.N = N
        self.resize_N = resize_N
        if effective_N is None:
            self.effective_N = N
        else:
            self.effective_N = effective_N
        self.transforms = self._get_aug(transforms_json)
        self.mean = mean
        self.std = std
        self.zoom_level = zoom_level
        self.output_type = output_type
        self.label_smoothing = label_smoothing
        self.use_saved = use_saved
        self.custom_normalization = custom_normalization

    def _get_aug(self, arg):
        with open(arg) as f:
            augs = A.from_dict(json.load(f))
        target = {}
        for i in range(1, self.N):
            target['image' + str(i)] = 'image'
        return A.Compose(augs, p=1, additional_targets=target)

    def __len__(self):
        return(len(self.data_df))

    def __getitem__(self, idx):
        """Will load the mask, get random coordinates around/with the mask,
        load the image by coordinates
        """
        slide_id = self.data_df.image_id.values[idx]
        isup_grade = self.data_df.isup_grade.values[idx]
        if not self.use_saved:
            mask = openslide.OpenSlide(os.path.join(
                self.mask_dir,
                f'{slide_id}_mask.tiff'))
            image = openslide.OpenSlide(os.path.join(
                self.image_dir,
                f'{slide_id}.tiff'))
            mask = mask.read_region(
                (0, 0),
                self.zoom_level,
                mask.level_dimensions[self.zoom_level])
            img = image.read_region(
                (0, 0),
                self.zoom_level,
                image.level_dimensions[self.zoom_level])
            img = np.asarray(img)[..., :3]
            mask = np.asarray(mask)[..., :3]
            tiles = tile(img, mask=mask, N=self.N, sz=self.crop_size)
            tiled_images = tiles['img']
        else:
            input_file = os.path.join(
                self.image_dir,
                f"{slide_id}_{self.zoom_level}_{self.N}_{self.crop_size}.pkl")
            with open(input_file, "rb") as input_file:
                tiled_images = pickle.load(input_file)
        target_names = ['image' + str(i) if i > 0 else 'image'
                        for i in range(len(tiled_images))]
        tiled_images = dict(zip(
            target_names,
            tiled_images))
        augmented = self.transforms(**tiled_images)
        tiled_images = [augmented[target] for target in target_names]
        tiled_images = tiled_images[:self.effective_N]
        if self.resize_N is not None:
            tiled_images = [A.Resize(self.resize_N, self.resize_N)(image=image)['image']
                            for image in tiled_images]
        # TO DO: randomly select patches from pre-loaded
        tiled_images = np.stack(tiled_images)
        if self.custom_normalization:
            tiled_images = (1.0 - tiled_images/255.0)
            tiled_images = (tiled_images - self.mean)/self.std
        assert len(tiled_images) == self.effective_N
        tiled_images = tiled_images.transpose(0, 3, 1, 2)
        # Fix outputs for each of the tasks
        # To do: move as separate function
        if self.output_type == 'regression':
            isup_grade = np.expand_dims(isup_grade, 0)
            isup_grade = torch.from_numpy(isup_grade).float()
        elif self.output_type == 'classification':
            isup_grade = torch.tensor(isup_grade)
        elif self.output_type == 'ordinal':
            raise NotImplementedError
        elif self.output_type == 'ohe_classification':
            '''
            We can make it asymetric if we use the assumption that
            errors 4-5 are more likely than errors 0-5
            '''
            ohe_isup_grade = np.zeros(6)
            ohe_isup_grade[isup_grade] = 1
            isup_grade = ohe_isup_grade
            isup_grade = isup_grade * (1 - self.label_smoothing) + \
                self.label_smoothing / 6
            isup_grade = torch.from_numpy(isup_grade).float()
        else:
            raise NotImplementedError

        data = {'features': torch.from_numpy(tiled_images).float(),
                'targets': isup_grade}
        return(data)


class ClassifcationDatasetMultiCropMultiHead(Dataset):
    def __init__(
        self,
        data_df,
        transforms_json,
        image_dir,
        mask_dir,
        mean=np.array([0.90949707, 0.8188697, 0.87795304]),
        std=np.array([0.36357649, 0.49984502, 0.40477625]),
        N=16,
        zoom_level=2,
        crop_size=128,
        label_smoothing=0.15,
        output_type='classification',
        use_saved=False,
        *args,
            **kwargs):
        """Prepares pytorch dataset for training
        Generates tiles from coarse slide and returns it

        Args:
            data_df (pd.DataFrame): data.frame with slides id and labels.
            augmentations (albumentations.compose): augmentations.
            image_dir (str): folder with images.
            mask_dir (str): folder with masks.
            crop_size(int): crop size around mask. Default: 128
        Returns
            Dataset

        """
        self.data_df = pd.read_csv(data_df)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.crop_size = crop_size
        self.N = N
        self.transforms = self._get_aug(transforms_json)
        self.mean = mean
        self.std = std
        self.zoom_level = zoom_level
        self.output_type = output_type
        self.label_smoothing = label_smoothing
        self.use_saved = use_saved

    def _get_aug(self, arg):
        with open(arg) as f:
            augs = A.from_dict(json.load(f))
        target = {}
        for i in range(1, self.N):
            target['image' + str(i)] = 'image'
        return A.Compose(augs, p=1, additional_targets=target)

    def __len__(self):
        return(len(self.data_df))

    def __getitem__(self, idx):
        """Will load the mask, get random coordinates around/with the mask,
        load the image by coordinates
        """
        slide_id = self.data_df.image_id.values[idx]
        isup_grade = self.data_df.isup_grade.values[idx]
        gleason = self.data_df.gleason_score.values[idx]
        # Dirty way to prepare gleason score
        gleason_to_number = {
            '0': 0,
            '3': 1,
            '4': 2,
            '5': 3
        }
        if gleason == 'negative':
            gleason_major = 0
            gleason_minor = 0
        else:
            gleason_major = gleason.split('+')[0]
            gleason_major = gleason_to_number[gleason_major]
            gleason_minor = gleason.split('+')[1]
            gleason_minor = gleason_to_number[gleason_minor]
        if not self.use_saved:
            mask = openslide.OpenSlide(os.path.join(
                self.mask_dir,
                f'{slide_id}_mask.tiff'))
            image = openslide.OpenSlide(os.path.join(
                self.image_dir,
                f'{slide_id}.tiff'))
            mask = mask.read_region(
                (0, 0),
                self.zoom_level,
                mask.level_dimensions[self.zoom_level])
            img = image.read_region(
                (0, 0),
                self.zoom_level,
                image.level_dimensions[self.zoom_level])
            img = np.asarray(img)[..., :3]
            mask = np.asarray(mask)[..., :3]
            tiles = tile(img, mask=mask, N=self.N, sz=self.crop_size)
            tiled_images = tiles['img']
        else:
            input_file = os.path.join(
                self.image_dir,
                f"{slide_id}_{self.zoom_level}_{self.N}_{self.crop_size}.pkl")
            with open(input_file, "rb") as input_file:
                tiled_images = pickle.load(input_file)
        target_names = ['image' + str(i) if i > 0 else 'image'
                        for i in range(len(tiled_images))]
        tiled_images = dict(zip(
            target_names,
            tiled_images))
        augmented = self.transforms(**tiled_images)
        tiled_images = [augmented[target] for target in target_names]
        tiled_images = np.stack(tiled_images)
        tiled_images = (1.0 - tiled_images/255.0)
        tiled_images = (tiled_images - self.mean)/self.std
        assert len(tiled_images) == self.N
        tiled_images = tiled_images.transpose(0, 3, 1, 2)
        # Fix outputs for each of the tasks
        # To do: move as separate function
        if self.output_type == 'regression':
            isup_grade = np.expand_dims(isup_grade, 0)
            isup_grade = torch.from_numpy(isup_grade).float()
            gleason_major = np.expand_dims(gleason_major, 0)
            gleason_major = torch.from_numpy(gleason_major).float()
            gleason_minor = np.expand_dims(gleason_minor, 0)
            gleason_minor = torch.from_numpy(gleason_minor).float()
        elif self.output_type == 'classification':
            isup_grade = torch.tensor(isup_grade)
            gleason_major = torch.tensor(gleason_major)
            gleason_minor = torch.tensor(gleason_minor)
        elif self.output_type == 'ordinal':
            raise NotImplementedError
        elif self.output_type == 'ohe_classification':
            '''
            We can make it asymetric if we use the assumption that
            errors 4-5 are more likely than errors 0-5
            '''
            ohe_isup_grade = np.zeros(6)
            ohe_gleason_major = np.zeros(4)
            ohe_gleason_minor = np.zeros(4)

            ohe_isup_grade[isup_grade] = 1
            ohe_gleason_major[gleason_major] = 1
            ohe_gleason_minor[gleason_minor] = 1

            isup_grade = ohe_isup_grade
            gleason_major = ohe_gleason_major
            gleason_minor = ohe_gleason_minor

            isup_grade = isup_grade * (1 - self.label_smoothing) + \
                self.label_smoothing / 6
            gleason_major = gleason_major * (1 - self.label_smoothing) + \
                self.label_smoothing / 4
            gleason_minor = gleason_minor * (1 - self.label_smoothing) + \
                self.label_smoothing / 4

            isup_grade = torch.from_numpy(isup_grade).float()
            gleason_major = torch.from_numpy(gleason_major).float()
            gleason_minor = torch.from_numpy(gleason_minor).float()
        else:
            raise NotImplementedError

        data = {
            'features': torch.from_numpy(tiled_images).float(),
            'targets_isup': isup_grade,
            'targets_gleason_major': gleason_major,
            'targets_gleason_minor': gleason_minor,
            }
        return(data)


class MetricLearningAndClassifcationDatasetMultiCrop(Dataset):
    def __init__(
        self,
        data_df,
        transforms_json,
        image_dir,
        mean=np.array([0.90949707, 0.8188697, 0.87795304]),
        std=np.array([0.36357649, 0.49984502, 0.40477625]),
        N=16,
        zoom_level=2,
        crop_size=128,
        label_smoothing=0.15,
        output_type='classification',
        use_saved=False,
        *args,
            **kwargs):
        """Prepares pytorch dataset for training
        Generates tiles from coarse slide and returns it

        Args:
            data_df (pd.DataFrame): data.frame with slides id and labels.
            augmentations (albumentations.compose): augmentations.
            image_dir (str): folder with images.
            mask_dir (str): folder with masks.
            crop_size(int): crop size around mask. Default: 128
        Returns
            Dataset

        """
        self.data_df = pd.read_csv(data_df)
        self.image_dir = image_dir
        self.crop_size = crop_size
        self.N = N
        self.transforms = self._get_aug(transforms_json)
        self.mean = mean
        self.std = std
        self.zoom_level = zoom_level
        self.output_type = output_type
        self.label_smoothing = label_smoothing
        self.targets = self.data_df.isup_grade.unique().astype(str)
        self.use_saved = use_saved

    def _get_aug(self, arg):
        with open(arg) as f:
            augs = A.from_dict(json.load(f))
        target = {}
        for i in range(1, self.N):
            target['image' + str(i)] = 'image'
        return A.Compose(augs, p=1, additional_targets=target)

    def __len__(self):
        return(len(self.data_df))

    def __getitem__(self, idx):
        """Will load the mask, get random coordinates around/with the mask,
        load the image by coordinates
        """
        slide_id = self.data_df.image_id.values[idx]
        isup_grade = self.data_df.isup_grade.values[idx]
        isup_grade_class = self.data_df.isup_grade.values[idx]
        if not self.use_saved:
            image_openslide = openslide.OpenSlide(os.path.join(
                self.image_dir,
                f'{slide_id}.tiff'))
            img = image_openslide.read_region(
                (0, 0),
                self.zoom_level,
                image_openslide.level_dimensions[self.zoom_level])
            image_openslide.close()
            img = np.asarray(img)[..., :3]
            tiles = tile(img, N=self.N, sz=self.crop_size)
            del img
            gc.collect()
            tiled_images = tiles['img']
        else:
            input_file = os.path.join(
                self.image_dir,
                f"{slide_id}_{self.zoom_level}_{self.N}_{self.crop_size}.pkl")
            with open(input_file, "rb") as input_file:
                tiled_images = pickle.load(input_file)
        target_names = ['image' + str(i) if i > 0 else 'image'
                        for i in range(len(tiled_images))]
        tiled_images = dict(zip(
            target_names,
            tiled_images))
        augmented = self.transforms(**tiled_images)
        tiled_images = [augmented[target] for target in target_names]
        tiled_images = np.stack(tiled_images)
        tiled_images = (1.0 - tiled_images/255.0)
        tiled_images = (tiled_images - self.mean)/self.std
        assert len(tiled_images) == self.N
        tiled_images = tiled_images.transpose(0, 3, 1, 2)
        # Fix outputs for each of the tasks
        # To do: move as separate function
        if self.output_type == 'regression':
            isup_grade = np.expand_dims(isup_grade, 0)
            isup_grade = torch.from_numpy(isup_grade).float()
        elif self.output_type == 'classification':
            isup_grade = torch.tensor(isup_grade)
        elif self.output_type == 'ordinal':
            raise NotImplementedError
        elif self.output_type == 'ohe_classification':
            '''
            We can make it asymetric if we use the assumption that
            errors 4-5 are more likely than errors 0-5
            '''
            ohe_isup_grade = np.zeros(6)
            ohe_isup_grade[isup_grade] = 1
            isup_grade = ohe_isup_grade
            isup_grade = isup_grade * (1 - self.label_smoothing) + \
                self.label_smoothing / 6
            isup_grade = torch.from_numpy(isup_grade).float()
        else:
            raise NotImplementedError

        data = {'features': torch.from_numpy(tiled_images).float(),
                'targets': isup_grade,
                'metric_learning_targets': isup_grade_class}
        return(data)


class ClassifcationMultiCropInferenceDataset(Dataset):
    def __init__(
        self,
        data_df,
        transforms,
        image_dir,
        mean=np.array([0.90949707, 0.8188697, 0.87795304]),
        std=np.array([0.36357649, 0.49984502, 0.40477625]),
        N=16,
        zoom_level=2,
        crop_size=128,
        use_saved=False,
        *args,
            **kwargs):
        """Prepares pytorch dataset for training
        Generates tiles from coarse slide and returns it

        Args:
            data_df (pd.DataFrame): data.frame with slides id and labels.
            augmentations (albumentations.compose): augmentations.
            image_dir (str): folder with images.
            mask_dir (str): folder with masks.
            crop_size(int): crop size around mask. Default: 128
        Returns
            Dataset

        """
        self.data_df = pd.read_csv(data_df)
        self.image_dir = image_dir
        self.crop_size = crop_size
        self.N = N
        self.transforms = transforms
        self.mean = mean
        self.std = std
        self.zoom_level = zoom_level
        self.use_saved = use_saved

    def __len__(self):
        return(len(self.data_df))

    def __getitem__(self, idx):
        """Will load the mask, get random coordinates around/with the mask,
        load the image by coordinates
        """
        slide_id = self.data_df.image_id.values[idx]
        if not self.use_saved:
            image = openslide.OpenSlide(os.path.join(
                self.image_dir,
                f'{slide_id}.tiff'))
            img = image.read_region(
                (0, 0),
                self.zoom_level,
                image.level_dimensions[self.zoom_level])
            img = np.asarray(img)[..., :3]
            tiles = tile(img, N=self.N, sz=self.crop_size)
            tiled_images = tiles['img']
        else:
            input_file = os.path.join(
                self.image_dir,
                f"{slide_id}_{self.zoom_level}_{self.N}_{self.crop_size}.pkl")
            with open(input_file, "rb") as input_file:
                tiled_images = pickle.load(input_file)
        tiled_images = (1.0 - tiled_images/255.0)
        tiled_images = (tiled_images - self.mean)/self.std
        assert len(tiled_images) == self.N
        if self.transforms is not None:
            target_names = ['image' + str(i) if i > 0 else 'image'
                            for i in range(len(tiled_images))]
            tiled_images = dict(zip(
                target_names,
                tiled_images))
            augmented = self.transforms(**tiled_images)
            tiled_images = [augmented[target] for target in target_names]

        tiled_images = np.stack(tiled_images).transpose(0, 3, 1, 2)
        data = {'features': torch.from_numpy(tiled_images).float()}
        return(data)


class ClassifcationDatasetMultiCropOneImage(Dataset):
    def __init__(
        self,
        df,
        image_size,
        image_folder,
        level,
        n_tiles=36,
        tile_mode=0,
        label_smoothing=0.15,
        output_type='ordinal',
        normalize=True,
        rand=False,
        transform_individual=None,
        transform_global=None,
        load_pickled_tiles=False,
        pickled_tiles_folder=None
            ):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.image_folder = image_folder
        self.n_tiles = n_tiles
        self.tile_mode = tile_mode
        self.rand = rand
        self.transform_individual = transform_individual
        self.transform_global = transform_global
        self.level = level
        self.label_smoothing = label_smoothing
        self.output_type = output_type
        self.load_pickled_tiles = load_pickled_tiles
        self.pickled_tiles_folder = pickled_tiles_folder
        self.normalize = normalize

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row.image_id
        if self.load_pickled_tiles:
            pickled_image = f"{img_id}_{self.level}_{self.n_tiles}_{self.image_size}.pkl"
            with open(os.path.join(self.pickled_tiles_folder, pickled_image), 'rb') as f_:
                tiles = pickle.load(f_)
        else:
            tiff_file = os.path.join(self.image_folder, f'{img_id}.tiff')
            image = openslide.OpenSlide(tiff_file)
            image = image.read_region(
                (0, 0),
                self.level,
                image.level_dimensions[self.level])
            image = np.asarray(image)[..., :3]
            tiles, OK = tileV2(image, self.tile_mode, self.image_size, self.n_tiles)

        if self.rand:
            idxes = np.random.choice(list(range(self.n_tiles)), self.n_tiles, replace=False)
        else:
            idxes = list(range(self.n_tiles))

        n_row_tiles = int(np.sqrt(self.n_tiles))
        images = np.zeros((self.image_size * n_row_tiles, self.image_size * n_row_tiles, 3))
        for h in range(n_row_tiles):
            for w in range(n_row_tiles):
                i = h * n_row_tiles + w
                if len(tiles) > idxes[i]:
                    this_img = tiles[idxes[i]]['img']
                else:
                    this_img = np.ones((self.image_size, self.image_size, 3)).astype(np.uint8) * 255
                this_img = 255 - this_img
                if self.transform_individual is not None:
                    this_img = self.transform_individual(image=this_img)['image']
                h1 = h * self.image_size
                w1 = w * self.image_size
                images[h1:h1+self.image_size, w1:w1+self.image_size] = this_img

        if self.transform_global is not None:
            images = self.transform_global(image=images)['image']
        images = images.astype(np.float32)
        if self.normalize:
            images /= 255
        images = images.transpose(2, 0, 1)
        gleason = row.gleason_score
        # Dirty way to prepare gleason score
        gleason_to_number = {
            '0': 0,
            '3': 3,
            '4': 4,
            '5': 5
        }
        if gleason == 'negative':
            gleason_major = 0
            gleason_minor = 0
        else:
            gleason_major = gleason.split('+')[0]
            gleason_major = gleason_to_number[gleason_major]
            gleason_minor = gleason.split('+')[1]
            gleason_minor = gleason_to_number[gleason_minor]
        # Now return the label format we need
        if self.output_type == 'regression':
            isup_grade = np.expand_dims(row.isup_grade, 0)
            isup_grade = torch.from_numpy(isup_grade).float()
            gleason_major = np.expand_dims(gleason_major, 0)
            gleason_major = torch.from_numpy(gleason_major).float()
            gleason_minor = np.expand_dims(gleason_minor, 0)
            gleason_minor = torch.from_numpy(gleason_minor).float()
        elif self.output_type == 'classification':
            isup_grade = torch.tensor(row.isup_grade)
            gleason_major = torch.tensor(gleason_major)
            gleason_minor = torch.tensor(gleason_minor)
        elif self.output_type == 'ordinal':
            isup_grade = np.zeros(5).astype(np.float32)
            isup_grade[:row.isup_grade] = 1
            gleason_major_ord = np.zeros(5).astype(np.float32)
            gleason_major_ord[:gleason_major] = 1
            gleason_major = gleason_major_ord
            gleason_minor_ord = np.zeros(5).astype(np.float32)
            gleason_minor_ord[:gleason_minor] = 1
            gleason_minor = gleason_minor_ord
            gleason_minor = torch.from_numpy(gleason_minor).float()
            gleason_major = torch.from_numpy(gleason_major).float()
            isup_grade = torch.from_numpy(isup_grade).float()
        elif self.output_type == 'ohe_classification':
            '''
            We can make it asymetric if we use the assumption that
            errors 4-5 are more likely than errors 0-5
            '''
            ohe_isup_grade = np.zeros(6)
            ohe_gleason_major = np.zeros(6)
            ohe_gleason_minor = np.zeros(6)

            ohe_isup_grade[row.isup_grade] = 1
            ohe_gleason_major[gleason_major] = 1
            ohe_gleason_minor[gleason_minor] = 1

            isup_grade = ohe_isup_grade
            gleason_major = ohe_gleason_major
            gleason_minor = ohe_gleason_minor

            isup_grade = isup_grade * (1 - self.label_smoothing) + \
                self.label_smoothing / 6
            gleason_major = gleason_major * (1 - self.label_smoothing) + \
                self.label_smoothing / 6
            gleason_minor = gleason_minor * (1 - self.label_smoothing) + \
                self.label_smoothing / 6

            isup_grade = torch.from_numpy(isup_grade).float()
            gleason_major = torch.from_numpy(gleason_major).float()
            gleason_minor = torch.from_numpy(gleason_minor).float()
        else:
            raise NotImplementedError

        data = {
            'features': torch.from_numpy(images).float(),
            'targets_isup': isup_grade,
            'targets_gleason_major': gleason_major,
            'targets_gleason_minor': gleason_minor,
            }
        return(data)
