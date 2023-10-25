#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/11 16:43
# @Author  : zhangchao
# @File    : trainer.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
import os
import os.path as osp
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import random_split, DataLoader

from EsophagealCancer.external.unet.model import UNet
from EsophagealCancer.loss import dice_loss, dice_coeff, multiclass_dice_coeff
from EsophagealCancer.utils import BaseDataset, get_format_time


def train_model(
        image_list,
        n_classes,
        bilinear=False,
        val_percent=0.1,
        batch_size=1,
        epochs=100,
        learning_rate=1e-5,
        weight_decay=1e-8,
        momentum=0.999,
        gpu=0,
        amp=False,
        gradient_clipping=1.0,
        save_checkpoint=True,
        dir_checkpoint='./checkpoints'):
    # 1. Create Dataset
    dataset = BaseDataset(image_path_list=image_list,
                          image_suffix='equalhist',
                          mask_suffix='label',
                          scale=1.)
    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset=dataset,
                                      lengths=[n_train, n_val],
                                      generator=torch.Generator().manual_seed(0))
    # 3. Create data loader
    # loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    # 4. Setup the optimizer, the loss, the learning rate scheduler and the scaling for AMP
    device = torch.device("cpu" if not torch.cuda.is_available() else f"cuda:{gpu}")
    model = UNet(n_channels=3, n_classes=n_classes, bilinear=bilinear)
    model.to(device)
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate,
                              weight_decay=weight_decay,
                              momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = torch.nn.CrossEntropyLoss() if model.n_classes > 1 else torch.nn.BCEWithLogitsLoss()
    global_step = 0

    for eph in range(epochs):
        model.train()
        for data in train_loader:
            image, label = data
            assert image.shape[1] == model.n_channels, \
                f'Network has been defined with {model.n_channels} input channels, ' \
                f'but loaded images have {image.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            label = label.to(device=device, dtype=torch.long)
            if label.size(1) == 1:
                label.squeeze_(1)

            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                pred = model(image)
                if model.n_classes == 1:
                    loss = criterion(pred.squeeze(1), label.float())
                    loss += dice_loss(F.sigmoid(pred.squeeze(1)), label.float(), multiclass=False)
                else:
                    loss = criterion(pred, label)
                    loss += dice_loss(
                        F.softmax(pred, dim=1).float(),
                        F.one_hot(label, model.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True
                    )
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            print(f"[{get_format_time()}] Epoch: {eph} Loss: {loss.item():4f}")
            global_step += 1

            division_step = (n_train // (5 * batch_size))
            if division_step > 0:
                if global_step % division_step == 0:
                    val_score = evaluate(model, val_loader, device, amp)
                    scheduler.step(val_score)
                    print(f"[{get_format_time()}] validation dice score: {val_score}")

        if save_checkpoint:
            os.makedirs(dir_checkpoint, exist_ok=True)
            torch.save(model, osp.join(dir_checkpoint, f'unet_checkpoint.bin'))


def evaluate(model, data_loader, device, amp):
    model.eval()
    dice_score = 0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for data in data_loader:
            image, label = data

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            label = label.to(device=device, dtype=torch.long)

            # predict the mask
            pred = model(image)

            if model.n_classes == 1:
                assert label.min() >= 0 and label.max() <= 1, 'True mask indices should be in [0, 1]'
                pred = (F.sigmoid(pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(pred, label, reduce_batch_first=False)
            else:
                assert label.min() >= 0 and label.max() < model.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                label = F.one_hot(label, model.n_classes).permute(0, 3, 1, 2).float()
                pred = F.one_hot(pred.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(pred[:, 1:], label[:, 1:], reduce_batch_first=False)

    model.train()
    return dice_score / max(len(data_loader), 1)



