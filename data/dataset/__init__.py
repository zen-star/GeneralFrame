# -- coding: utf-8 --
"""
@author: Zheng Shida
@contact: luckyzsd@163.com
"""

from .coco import Coco
from .hikhandarm import HikHandArm

__factory = {
    'HikHandArm': HikHandArm,
    'coco': Coco,
    # 'MHPv2': MHPv2,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError('Unknown dataset: {}'.format(name))
    return __factory[name](*args, **kwargs)
