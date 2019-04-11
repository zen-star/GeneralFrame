# -- coding:utf-8 --
"""
@author: Zheng Shida
@contact: luckyzsd@163.com
"""
from PIL import Image
import numpy as np
import random
import math
# --------------------------------------------------------------------------------------------------------------
# The following methods are all has dict input/output
# e.g.
# sample['imgage'] = raw RGB input image
# sample['seg_label'] = groun_truth_semantic_segmentation_image (values in [0,1,2,...,n_classes])
# sample['inst_label'] = ground_truth_instance_image_list[inst_img1,inst_img2,...,inst_imgX] (values in [0,1])
# --------------------------------------------------------------------------------------------------------------


class Normalize(object):
    """
    Normalize a
    """


class ToTensor(object):
    """
    Convert numpy.array to
    """


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.2, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = [int(round(i*255)) for i in mean]
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        """
        :param img: PIL.Image, img.size = w,h
        :return: PIL.Image
        """
        if random.uniform(0, 1) >= self.probability:
            return img
        img = np.array(img).transpose((2, 0, 1))  # w,h -> h,w,c -> c,h,w
        for attempt in range(100):
            area = img.shape[1] * img.shape[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[2] and h < img.shape[1]:
                x1 = random.randint(0, img.shape[1] - h)
                y1 = random.randint(0, img.shape[2] - w)
                if img.shape[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                img = Image.fromarray(img.transpose((1, 2, 0)), mode='RGB')  # c,h,w -> h,w,c
                return img
        img = Image.fromarray(img.transpose((1, 2, 0)), mode='RGB')  # c,h,w -> h,w,c
        return img


if __name__ == '__main__':

    test_img = Image.open('../../test_images/ycy.jpg')
    test_random_erase = RandomErasing(probability=0.9)
    after_img = test_random_erase(test_img)
    after_img.show()
