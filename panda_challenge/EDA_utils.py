import openslide
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
from .utils import crop_around_mask


def plot_count(
    df,
    feature,
    ax,
    title='',
    size=2,
    *args,
        **kwargs):
    # source : kaggle.com/rohitsingh9990/panda-eda-better-visualization-simple-baseline
    total = float(len(df))
    sns.countplot(
        df[feature],
        order=df[feature].value_counts().index,
        palette='Set2',
        ax=ax,
        *args,
        **kwargs)
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center")


def plot_relative_distribution(
    df,
    feature,
    hue,
    ax,
    title='',
    size=2,
    *args,
        **kwargs):
    # source : kaggle.com/rohitsingh9990/panda-eda-better-visualization-simple-baseline
    total = float(len(df))
    sns.countplot(
        x=feature,
        hue=hue,
        data=df,
        palette='Set2',
        *args,
        **kwargs)
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center")


def makeSimpleEDAplots(df):
    f, ax = plt.subplots(2, 2, figsize=(15, 9))
    ax = ax.ravel()
    plot_count(
        df=df,
        feature='data_provider',
        title='data_provider count and % plot',
        ax=ax[0])
    plot_count(
        df=df,
        feature='isup_grade',
        title='isup_grade count and %age plot',
        ax=ax[1])
    plot_count(
        df=df,
        feature='gleason_score',
        title='gleason_score count and %age plot',
        size=3,
        ax=ax[2])
    plot_relative_distribution(
        df=df,
        feature='gleason_score',
        hue='data_provider',
        title='relative count plot of gleason_score with data_provider',
        size=3,
        ax=ax[3])
    plt.tight_layout()


def visualizeExampleSingeSlide(
    slide_id,
    ax,
    data_provider,
    isup_grade,
    gleason_score,
    DATA_DIR,
    *args,
        **kwargs):
    """
    source: https://www.kaggle.com/rohitsingh9990/panda-eda-better-visualization-simple-baseline
    """
    image = openslide.OpenSlide(os.path.join(
        DATA_DIR,
        f'train_images/{slide_id}.tiff'))
    mask = openslide.OpenSlide(os.path.join(
        DATA_DIR,
        f'train_label_masks/{slide_id}_mask.tiff'))
    image_patch = image.read_region(
        (0, 0),
        image.level_count - 1,
        image.level_dimensions[-1])
    mask_patch = mask.read_region(
        (0, 0),
        mask.level_count - 1,
        mask.level_dimensions[-1])
    ax.imshow(np.asarray(image_patch)[..., :3], *args, **kwargs)
    ax.imshow(np.asarray(mask_patch)[..., 0], alpha=0.25, *args, **kwargs)
    image_id = slide_id
    ax.set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")


def visualizeExampleMaskSlide(
    slide_id,
    ax,
    isup_grade,
    data_provider,
    gleason_score,
    DATA_DIR,
    *args,
        **kwargs):
    """
    source: https://www.kaggle.com/rohitsingh9990/panda-eda-better-visualization-simple-baseline
    """
    mask = openslide.OpenSlide(os.path.join(
        DATA_DIR,
        f'train_label_masks/{slide_id}_mask.tiff'))
    cmap = matplotlib.colors.ListedColormap([
        'black',
        'gray',
        'green',
        'yellow',
        'orange',
        'red'])
    mask_patch = mask.read_region(
        (0, 0),
        mask.level_count - 1,
        mask.level_dimensions[-1])
    ax.imshow(
        np.asarray(mask_patch)[..., 0],
        cmap=cmap,
        interpolation='nearest',
        vmin=0,
        vmax=5,
        *args,
        **kwargs)
    image_id = slide_id
    ax.set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")


def get_desc_image(df, slide_id):
    dp = df.data_provider[df.image_id == slide_id].values[0]
    isup_g = df.isup_grade[df.image_id == slide_id].values[0]
    gl_s = df.gleason_score[df.image_id == slide_id].values[0]
    return dp, isup_g, gl_s


def cropRandomExample(slide_id, DATA_DIR, cropsize=512):
    f, ax = plt.subplots(1, 3)
    mask = openslide.OpenSlide(os.path.join(
        DATA_DIR,
        f'train_label_masks/{slide_id}_mask.tiff'))
    image = openslide.OpenSlide(os.path.join(
        DATA_DIR,
        f'train_images/{slide_id}.tiff'))
    k = int(image.level_downsamples[-1])
    mask_full = mask.read_region((0, 0), 2, mask.level_dimensions[-1])
    mask_full = np.asarray(mask_full).astype(bool)[..., 0]
    ax[0].imshow(mask_full)
    crop_coords = crop_around_mask(mask_full, 32, 32)
    image_slice = image.read_region(
        (k*crop_coords['x_min'], k*crop_coords['y_min']),
        0,
        (cropsize, cropsize))
    mask_slice = mask.read_region(
        (k*crop_coords['x_min'], k*crop_coords['y_min']),
        0,
        (cropsize, cropsize))
    ax[1].imshow(np.asarray(image_slice)[..., :3])
    ax[1].imshow(np.asarray(mask_slice)[..., 0], alpha=0.5)
    cmap = matplotlib.colors.ListedColormap([
        'black',
        'gray',
        'green',
        'yellow',
        'orange',
        'red'])
    ax[2].imshow(
        np.asarray(mask_slice)[..., 0],
        cmap=cmap,
        interpolation='nearest',
        vmin=0,
        vmax=5)
    plt.tight_layout()
