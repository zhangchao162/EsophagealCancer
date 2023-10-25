#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/11 16:56
# @Author  : zhangchao
# @File    : dataset.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
import os.path as osp
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from EsophagealCancer.utils import get_format_time


class BaseDataset(Dataset):
    def __init__(self, image_path_list, image_suffix='image', mask_suffix='label', scale=1.):
        super(BaseDataset, self).__init__()
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1., '`scale` must be between 0 and 1.'
        self.scale = scale

        self.images = [file for file in image_path_list if osp.isfile(file)]

        print(f"[{get_format_time()}] Creating dataset with {self.__len__()} examples.")
        print(f"[{get_format_time()}] Scanning mask files to determine unique values.")
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        data = Image.open(self.images[idx]).convert("RGB")
        label = Image.open(
            self.images[idx].replace(self.image_suffix, self.mask_suffix).replace("jpg", "png")).convert("L")
        # label = cv2.imread(self.images[idx].replace(self.image_suffix, self.mask_suffix), flags=0)
        data = self.transform(data)
        label = self.transform(label)
        return data, label


