# -- coding:utf-8 --
"""
@author:  Zheng Shida
@contact: luckyzsd@163.com
"""

import torchvision.transforms as T

from .transform import RandomErasing


def build_transforms(cfg, is_train=True):
    if cfg.DATASET_NAME == 'coco':
        if is_train:
            transform = T.Compose([
                T.Resize(cfg.SIZE_TRAIN),
                T.RandomHorizontalFlip(p=cfg.FLIP_PROB),
                T.Pad(cfg.PAD_SIZE),
                T.RandomCrop(cfg.SIZE_TRAIN),
                T.ToTensor(),
                T.Normalize(mean=cfg.IMAGE_MEAN, std=cfg.IMAGE_STD),
                RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.IMAGE_MEAN)
            ])
        else:
            transform = T.Compose([
                T.Resize(cfg.SIZE_TEST),
                T.ToTensor(),
                T.Normalize(mean=cfg.IMAGE_MEAN, std=cfg.IMAGE_STD)
            ])

    elif cfg.DATASET_NAME == 'HikHandArm':
        if is_train:
            transform = T.Compose([
                T.Resize(cfg.SIZE_TRAIN),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.Pad(cfg.INPUT.PADDING),
                T.RandomCrop(cfg.SIZE_TRAIN),
                T.ToTensor(),
                T.Normalize(mean=cfg.IMAGE_MEAN, std=cfg.IMAGE_STD),
                RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.IMAGE_MEAN)
            ])
        else:
            transform = T.Compose([
                T.Resize(cfg.SIZE_TEST),
                T.ToTensor(),
                T.Normalize(mean=cfg.IMAGE_MEAN, std=cfg.IMAGE_STD)
            ])

    else:
        print('Wrong input dataset name, now only support {}.'.format(cfg.SUPPORT_DATASET))
        raise NotImplementedError

    return transform
