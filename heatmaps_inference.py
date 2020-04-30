import os
import pandas as pd
import numpy as np
import json
from panda_challenge.train_utils import get_model
from panda_challenge.inference_utils import get_slice_heatmap
import torch
import openslide
from tqdm.auto import tqdm


if __name__ == '__main__':
    with open('./configs/test_run_v2_config.json', 'r') as f:
        params = json.load(f)    
    data_val = pd.read_csv('/data/personal_folders/skolchenko/panda/data_val.csv')
    model = get_model(
        params['model_name'],
        **params['model_config'])
    model.cuda()
    checkpoint = torch.load(os.path.join(params['log_dir'], 'checkpoints/best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()   
    # Save predicted heatmaps
    all_cls_lbl_distr = []
    for idx in tqdm(range(len(data_val))):
        slide_id = data_val.image_id.values[idx]
        isup_grade = data_val.isup_grade.values[idx]
        mask = openslide.OpenSlide(os.path.join(
            params['val_mask_dir'],
            f'{slide_id}_mask.tiff'))
        image = openslide.OpenSlide(os.path.join(
            params['val_image_dir'],
            f'{slide_id}.tiff'))  
        predictions = get_slice_heatmap(slide_id, params['val_image_dir'], model=model, normalize=False, bsize=48)
        np.save(f'/data/personal_folders/skolchenko/panda/prediction_heatmaps/{slide_id}.npy', predictions)   