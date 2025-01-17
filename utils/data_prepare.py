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


def gray2DLRSDcolor():  # 将灰度mask图转换为彩色标签色块图展示
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

    mask_dir = '/media/D_4TB/YL_4TB/Segmentation/data/DLRSD/masks'

    mask_files = glob.glob(mask_dir + '/**/*.npy')
    color_rgb_np = np.array(color_rgb_list).astype(np.uint8)
    color_map = color_rgb_np.copy()
    color_map.resize((256, 3), refcheck=False)
    for mask_file in tqdm(mask_files):
        mask = np.load(mask_file)
        # image_display = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

        c1 = cv2.LUT(mask, color_map[:, 0])
        c2 = cv2.LUT(mask, color_map[:, 1])
        c3 = cv2.LUT(mask, color_map[:, 2])
        image_display = np.dstack((c1, c2, c3))
        display_file = mask_file.replace('/masks/', '/masks_display/').replace('.npy', '.png')
        dir = os.path.dirname(display_file)
        if not os.path.exists(dir):
            os.makedirs(dir)
        cv2.imwrite(display_file, cv2.cvtColor(image_display, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    # DLRSD_color2gray_mask()
    gray2DLRSDcolor()







