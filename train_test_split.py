from sklearn.model_selection import train_test_split
import pandas as pd
import os

if __name__ == '__main__':
    DATA_DIR = '/data/personal_folders/skolchenko/panda'
    IMAGE_DIR = '/data/personal_folders/skolchenko/panda/train_images/'
    MASK_DIR = '/data/personal_folders/skolchenko/panda/train_label_masks//'
    TRAIN_LABELS = '/data/personal_folders/skolchenko/panda/train.csv'
    train_labes = pd.read_csv(TRAIN_LABELS)
    has_slide = []
    for image_id in train_labes.image_id.values:
        has_slide.append(os.path.isfile(os.path.join(
            MASK_DIR,
            f'{image_id}_mask.tiff')))
    train_labes['has_slide'] = has_slide
    train_labes = train_labes.loc[train_labes['has_slide'], :]
    data_train, data_val = train_test_split(
        train_labes,
        test_size=0.25,
        random_state=42)
    data_train.to_csv(
        '/data/personal_folders/skolchenko/panda/data_train.csv',
        index=False)
    data_val.to_csv(
        '/data/personal_folders/skolchenko/panda/data_val.csv',
        index=False)
