#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :data_prepare.py
# @Time      :2021/11/8 下午7:07
# @Author    :Yangliang
import glob
import os

from cv2 import cv2
import numpy as np
from tqdm import tqdm


def DLRSD_color2gray_mask():    # 彩色标签图转灰度mask图函数 按照颜色表
    mask_images_dir = '/media/D_4TB/YL_4TB/Segmentation/data/DLRSD/masks_origin'
    image_files = glob.glob(mask_images_dir + '/**/*.png')
    color_rgb_list = [[0, 0, 0],
                      [166, 202, 240], [128, 128, 0], [0, 0, 128], [255, 0, 0], [0, 128, 0],
                      [128, 0, 0], [255, 233, 233], [160, 160, 164], [0, 128, 128], [90, 87, 255],
                      [255, 255, 0], [255, 192, 0], [0, 0, 255], [255, 0, 192], [128, 0, 128],
                      [0, 255, 0], [0, 255, 255]]
    classes = ['background',
               'airplane', 'bare soil', 'buildings', 'cars', 'chaparral',
               'court', 'dock', 'field', 'grass', 'mobile home',
               'pavement', 'sand', 'sea', 'ship', 'tanks',
               'trees', 'water']

    color_rgb_np = np.array(color_rgb_list)
    for image_file in tqdm(image_files):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        for i in range(len(color_rgb_list)):
            if i == 0: continue
            indexes = np.where((image[:, :, 0] == color_rgb_np[i][0]) & (image[:, :, 1] == color_rgb_np[i][1]) & (image[:, :, 2] == color_rgb_np[i][2]))
            mask[indexes[0], indexes[1]] = i
        mask_file = image_file.replace('/masks_origin/', '/masks/').replace('.png', '.npy')
        dir = os.path.dirname(mask_file)
        if not os.path.exists(dir):
            os.makedirs(dir)
        np.save(mask_file, mask)


if __name__ == "__main__":
    DLRSD_color2gray_mask()







