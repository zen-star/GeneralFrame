# -- coding:utf8 --
"""
@author: Zheng Shida
@contact: luckyzsd@163.com
"""
import numpy as np
from configs.config import config


def gen_mask_tensor2pil(semantic_score_map):
    """
    from semantic score map to mask for visualization
    :param semantic_score_map: torch.Tensor, model output semantic score map(after softmax), C,H,W
    :return: PIL.Image,
    """



def make_palette_27(num_classes):
    """
    create palette according to classes.
    :param num_classes: int, num of classes
    :return: palette
    """

    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for k in range(0, num_classes):
        label = k
        i = 0
        while label:
            palette[k, 0] |= (((label >> 0) & 1) << (7 - i))
            palette[k, 1] |= (((label >> 1) & 1) << (7 - i))
            palette[k, 2] |= (((label >> 2) & 1) << (7 - i))
            label >>= 3
            i += 1
    return palette
