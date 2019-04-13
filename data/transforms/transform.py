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

__all__ = ['Normalize', 'ToTensor', 'RandomHorizonFlip', 'RandomErasing', 'RandomVerticalFlip', 'Random90Rotate']


class Normalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']  # PIL.Image -> numpy.arrays(float32)
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


class RandomHorizonFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        img = sample['image']  # PIL.Image
        mask = sample['seg_label']  # numpy.arrays(uint8)
        inst = sample['inst_label']

        if random.random() < self.prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = Image.fromarray(mask.astype(np.uint8), mode='L').transpose(Image.FLIP_LEFT_RIGHT)
            mask = np.array(mask)
            for i in range(len(inst)):
                inst[i] = Image.fromarray(inst[i].astype(np.uint8), mode='L').transpose(Image.FLIP_LEFT_RIGHT)
                inst[i] = np.array(inst[i])

        return {
            'image': img,
            'seg_label': mask,
            'inst_label': inst
        }


class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        img = sample['image']  # PIL.Image
        mask = sample['seg_label']  # numpy.arrays(uint8)
        inst = sample['inst_label']

        if random.random() < self.prob:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = Image.fromarray(mask.astype(np.uint8), mode='L').transpose(Image.FLIP_TOP_BOTTOM)
            mask = np.array(mask)
            for i in range(len(inst)):
                inst[i] = Image.fromarray(inst[i].astype(np.uint8), mode='L').transpose(Image.FLIP_TOP_BOTTOM)
                inst[i] = np.array(inst[i])

        return {
            'image': img,
            'seg_label': mask,
            'inst_label': inst
        }


class Random90Rotate(object):
    def __init__(self, prob=0.5, right_or_left_prob=0.5):
        self.prob = prob
        self.rol_prob = right_or_left_prob

    def __call__(self, sample):
        img = sample['image']  # PIL.Image
        mask = sample['seg_label']  # numpy.arrays(uint8)
        inst = sample['inst_label']

        if random.random() < self.prob:

            def _rotate(img, mask, inst, method):
                img = img.transpose(method)
                mask = Image.fromarray(mask.astype(np.uint8), mode='L').transpose(method)
                mask = np.array(mask)
                inst2 = []
                for i in range(len(inst)):
                    inst2[i] = Image.fromarray(inst[i].astype(np.uint8), mode='L').transpose(method)
                    inst2[i] = np.array(inst2[i])
                return img, mask, inst2

            if random.random() < self.rol_prob:
                img, mask, inst = _rotate(img, mask, inst, Image.ROTATE_90)
            else:
                img, mask, inst = _rotate(img, mask, inst, Image.ROTATE_270)

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

    def __call__(self, sample):
        img = sample['image']  # PIL.Image
        mask = sample['seg_label']  # numpy.arrays(uint8)
        inst = sample['inst_label']
        if random.uniform(0, 1) >= self.probability:
            return sample
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
                mask[x1:x1 + h, y1:y1 + w] = 0.
                for i in range(len(inst)):
                    inst[i, x1:x1 + h, y1:y1 + w] = 0.
                if img.shape[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                    img = Image.fromarray(img.transpose((1, 2, 0)), mode='RGB')  # c,h,w -> h,w,c
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img = Image.fromarray(img, mode='L')  # c,h,w -> h,w,c
                return {
                    'image': img,
                    'seg_label': mask,
                    'inst_label': inst
                }
        return sample


if __name__ == '__main__':

    test_id_list = [5]  # according to __all__

    test_img = Image.open('../../test_images/ycy.jpg')
    mask = test_img.getchannel(0).point(lambda i: i < 200 and 255)
    # mask.show(title='seg_before')
    mask = np.array(mask)
    # Image.fromarray(mask, mode='L').show()
    mask1 = test_img.getchannel(0).point(lambda i: i < 100 and 255)
    # mask1.show(title='inst1_before')
    mask1 = np.array(mask1)
    mask2 = test_img.getchannel(0).point(lambda i: 100 <= i < 200 and 255)
    # mask2.show(title='inst2_before')
    mask2 = np.array(mask2)
    insts = np.concatenate((np.expand_dims(mask1, 0), np.expand_dims(mask2, 0)), 0)
    sample = {
        'image': test_img,
        'seg_label': mask,
        'inst_label': insts
    }

    if 3 in test_id_list:
        # test for random erase
        test_random_erase = RandomErasing(probability=0.9)
        after_sample = test_random_erase(sample)
        after_sample['image'].show(title='img_after')
        Image.fromarray(after_sample['seg_label'], mode='L').point(lambda i: i < 1 and 255).show(title='seg_after')
        Image.fromarray(after_sample['inst_label'][0], mode='L').point(lambda i: i < 1 and 255).show(title='inst_after')
        Image.fromarray(after_sample['inst_label'][1], mode='L').point(lambda i: i < 1 and 255).show(title='inst_after')
    else:
        after_sample = sample

    if 2 in test_id_list:
        # test for random horizon flip
        test_random_hflip = RandomHorizonFlip(0.9)
        after_sample = test_random_hflip(after_sample)
        after_sample['image'].show(title='img_after')
        Image.fromarray(after_sample['seg_label'], mode='L').point(lambda i: i < 1 and 255).show(title='seg_after')
        Image.fromarray(after_sample['inst_label'][0], mode='L').point(lambda i: i < 1 and 255).show(title='inst_after')
        Image.fromarray(after_sample['inst_label'][1], mode='L').point(lambda i: i < 1 and 255).show(title='inst_after')

    if 4 in test_id_list:
        # test for random horizon flip
        test_random_vflip = RandomVerticalFlip(0.9)
        after_sample = test_random_vflip(after_sample)
        after_sample['image'].show(title='img_after')
        Image.fromarray(after_sample['seg_label'], mode='L').point(lambda i: i < 1 and 255).show(title='seg_after')
        Image.fromarray(after_sample['inst_label'][0], mode='L').point(lambda i: i < 1 and 255).show(title='inst_after')
        Image.fromarray(after_sample['inst_label'][1], mode='L').point(lambda i: i < 1 and 255).show(title='inst_after')

    if 5 in test_id_list:
        # test for random horizon flip
        test_random_90 = Random90Rotate(0.9)
        after_sample = test_random_90(after_sample)
        after_sample['image'].show(title='img_after')
        Image.fromarray(after_sample['seg_label'], mode='L').point(lambda i: i < 1 and 255).show(title='seg_after')
        Image.fromarray(after_sample['inst_label'][0], mode='L').point(lambda i: i < 1 and 255).show(title='inst_after')
        Image.fromarray(after_sample['inst_label'][1], mode='L').point(lambda i: i < 1 and 255).show(title='inst_after')
