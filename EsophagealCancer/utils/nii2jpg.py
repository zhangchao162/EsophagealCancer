#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2023/10/19 22:03
# @Author : Chao Zhang
# @Email : 1623804006@qq.com
# @File : nii2jpg.py
import numpy as np
import nrrd
import os
import cv2
import uuid


def extract(conf, label_arr, src_arr, res_path):
    for i in range(conf["sizes"][2]):
        if label_arr[:, :, i].sum() != 0:
            label = label_arr[:, :, i]
            src = src_arr[:, :, i]
            file_name = uuid.uuid1()
            for j in np.unique(label.flatten()):
                if j == 0:
                    continue
                label_path = os.path.join(res_path, "label", case_id, f"{j}")
                img_path = os.path.join(res_path, "image", case_id, f"{j}")
                os.makedirs(label_path, exist_ok=True)
                os.makedirs(img_path, exist_ok=True)
                cv2.imwrite(os.path.join(label_path, f"{file_name}.png"), label * 255)
                cv2.imwrite(os.path.join(img_path, f"{file_name}.jpg"), src)


res_path = "img_dataset"
case_path = "C:\\Users\\Administrator\\Desktop\\鳞状食管癌"

for case_id in os.listdir(case_path):
    if case_id in os.listdir(os.path.join(res_path, "label")):
        continue

    src_path, label_path = None, None

    for file in [file for file in (os.listdir(os.path.join(case_path, case_id))) if file.endswith("nrrd")]:
        if "Segmentation" not in file:
            src_path = os.path.join(case_path, case_id, file)
        elif "label" in file:
            label_path = os.path.join(case_path, case_id, file)
        else:
            continue

    assert src_path is not None
    assert label_path is not None
    data = nrrd.read(label_path)
    data2 = nrrd.read(src_path)

    label_arr, conf = data
    src_arr, _ = data2

    extract(conf, label_arr, src_arr, res_path)

#
for case_id in os.listdir(case_path):
    if case_id in os.listdir(os.path.join(res_path, "equalhist")):
        continue
    for a in ["1", "2", "3"]:
        for file in os.listdir(os.path.join(res_path, "image", case_id, a)):
            img = cv2.imread(os.path.join(res_path, "image", case_id, a, file))

            res = cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)), cv2.COLOR_GRAY2RGB)

            os.makedirs(os.path.join(res_path, "equalhist", case_id, a), exist_ok=True)
            cv2.imwrite(os.path.join(res_path, "equalhist", case_id, a, file), res)
