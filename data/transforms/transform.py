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
# sample['image'] = raw 'RGB' PIL.Image
# sample['seg_label'] = gt_semantic_segmentation_images numpy.arrays (values in [0,1,2,...,n_classes])
# sample['inst_label'] = gt_instance_image_list[inst_img1,inst_img2,...,inst_imgX] numpy.arrays (values in [0,1])
#
# Note: PIL.Image: w,h;    numpy image: h,w,c;    torch image: c,h,w
# --------------------------------------------------------------------------------------------------------------


class Normalize(object):
    """
    Normalize
    """
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']  # PIL.Image -> numpy.arrays
        mask = sample['seg_label']  # not changed
        inst = sample['inst_label']  # not changed

        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {
            'image': img,
            'mask': mask,
            'inst': inst
        }


class ToTensor(object):
    """
    Convert PIL.image/numpy.array to torch.FloatTensor
    Not for sample['inst_label'], remaining numpy.arrays
    Always the last step of transformation.
    """
    def __call__(self, sample):
        img = sample['image']  # numpy.arrays -> torch.FloatTensor
        mask = sample['seg_label']  # numpy.arrays -> torch.FloatTensor
        inst = sample['inst_label']  # not changed

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {
            'image': img,
            'seg_label': mask,
            'inst_label': inst
        }


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
