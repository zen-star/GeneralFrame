# -- coding:utf-8 --
"""
@author: Zheng Shida
@contact: luckyzsd@163.com
"""

import torch
import torchvision.transforms as T
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random
import math
# --------------------------------------------------------------------------------------------------------------
# The following methods are all has dict input/output
# e.g.
# sample['image'] = raw 'RGB' PIL.Image
# sample['seg_label'] = gt_semantic_segmentation_images numpy.arrays (values in [0,1,2,...,n_classes])
# sample['inst_label'] = gt_instance_image_list[inst_img1,inst_img2,...,inst_imgX] list[numpy.arrays] (values in [0,1])
#
# Note: PIL.Image: w,h;    numpy image: h,w,c;    torch image: c,h,w
# --------------------------------------------------------------------------------------------------------------

__all__ = ['Normalize', 'ToTensor', 'RandomFlip', 'Random90Rotate', 'Resize', 'RandomPad', 'RandomCrop',
           'RandomErasing', 'ColorJitter']


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
    Not for sample['inst_label'], remaining list[numpy.arrays]
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


class RandomFlip(object):
    def __init__(self, prob=0.5, method=Image.FLIP_LEFT_RIGHT):
        self.prob = prob
        self.method = method

    def __call__(self, sample):
        img = sample['image']  # PIL.Image
        mask = sample['seg_label']  # numpy.arrays(uint8)
        inst = sample['inst_label']

        if random.random() < self.prob:
            img = img.transpose(self.method)
            mask = Image.fromarray(mask.astype(np.uint8), mode='L').transpose(self.method)
            mask = np.array(mask)
            inst2 = []
            for i in range(len(inst)):
                inst2.append(np.array(Image.fromarray(inst[i].astype(np.uint8), mode='L').transpose(self.method)))

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
                    inst2.append(np.array(Image.fromarray(inst[i].astype(np.uint8), mode='L').transpose(method)))
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


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.resize = T.Resize(self.size, self.interpolation)

    def __call__(self, sample):
        img = sample['image']  # PIL.Image
        mask = sample['seg_label']  # numpy.arrays(uint8)
        inst = sample['inst_label']

        img = self.resize(img)
        mask = self.resize(Image.fromarray(mask.astype(np.uint8), mode='L'))
        mask = np.array(mask)
        inst2 = []
        for i in range(len(inst)):
            inst2.append(np.array(self.resize(Image.fromarray(inst[i].astype(np.uint8), mode='L'))))
        return {
            'image': img,
            'seg_label': mask,
            'inst_label': inst
        }


class RandomPad(object):
    def __init__(self, size, mean=(0.4914, 0.4822, 0.4465)):
        size = random.randint(0, size)
        self.size = ((size, size), (size, size))
        self.fill = tuple([int(round(i*255)) for i in mean])
        self.pad = T.Pad(size, fill=self.fill)

    def __call__(self, sample):
        img = sample['image']  # PIL.Image
        mask = sample['seg_label']  # numpy.arrays(uint8)
        inst = sample['inst_label']

        img = self.pad(img)
        mask = np.pad(mask, self.size, 'constant')
        for i in range(len(inst)):
            inst[i] = np.pad(inst[i], self.size, 'constant')
        return {
            'image': img,
            'seg_label': mask,
            'inst_label': inst
        }


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(img, output_size):
        """
        get crop center and w/h
        :param img: PIL.Image
        :param output_size: (h, w)
        :return: i, j, h, w
        """
        w, h = img.size
        th, tw = output_size
        if w <= tw and h <= th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, sample):
        img = sample['image']  # PIL.Image
        mask = sample['seg_label']  # numpy.arrays(uint8)
        inst = sample['inst_label']
        print('in random crop:')
        print(img.size)
        print(mask.shape)
        print(inst[0].shape)
        i, j, h, w = self.get_params(img, self.size)

        img = img.crop((j, i, j+w, i+h))  # (left, upper, right, lower)
        mask = mask[i:i+h, j:j+w]
        for ii in range(len(inst)):
            inst[ii] = inst[ii][i:i+h, j:j+w]
        return {
            'image': img,
            'seg_label': mask,
            'inst_label': inst
        }


class RandomGaussBlur(object):
    def __init__(self, blur=5):
        self.blur = random.randint(0, blur)

    def __call__(self, sample):
        img = sample['image']  # PIL.Image
        mask = sample['seg_label']  # numpy.arrays(uint8)
        inst = sample['inst_label']
        img = img.filter(ImageFilter.GaussianBlur(self.blur))

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
    def __init__(self, probability=0.5, sl=0.02, sh=0.1, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
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
            area = img.shape[1] * img.shape[2]  # h,w

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[2] and h < img.shape[1]:
                x1 = random.randint(0, img.shape[1] - h)
                y1 = random.randint(0, img.shape[2] - w)
                mask[x1:x1 + h, y1:y1 + w] = 0.
                for i in range(len(inst)):
                    inst[i][x1:x1 + h, y1:y1 + w] = 0.
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


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')

    @staticmethod
    def _check_input(value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, (float, int)):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def _do_jitter(img, Brightness, Contrast, Color):  # color denotes saturation
        var_dict = dict(locals())  # locals() is python in-build function to obtain 'var':value dynamically
        trans = []
        for var in var_dict:
            if len(var) > 3:
                trans.append(var)
        # trans = ['Brightness', 'Contrast', 'Color']
        random.shuffle(trans)
        for one in trans:
            if var_dict[one] is not None:
                one_factor = random.uniform(var_dict[one][0], var_dict[one][1])
                enhancer = getattr(ImageEnhance, one)(img)
                img = enhancer.enhance(one_factor)
        return img

    def __call__(self, img):
        img = sample['image']  # PIL.Image
        mask = sample['seg_label']  # numpy.arrays(uint8)
        inst = sample['inst_label']
        img = self._do_jitter(img, self.brightness, self.contrast, self.saturation)
        return {
            'image': img,
            'seg_label': mask,
            'inst_label': inst
        }

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        return format_string


if __name__ == '__main__':

    # selectively comment not-for-test methods
    transform_test = T.Compose([
        Resize(512),
        RandomFlip(0.8, Image.FLIP_LEFT_RIGHT),
        RandomFlip(0.8, Image.FLIP_TOP_BOTTOM),
        RandomGaussBlur(10),
        ColorJitter(brightness=(0.25, 1.5), contrast=(0.5, 1.5), saturation=(0.75, 1.25)),
        Random90Rotate(0.8, 0.8),
        RandomErasing(probability=0.8),
        RandomPad(15),
        RandomCrop((512, 512)),
    ])
    # prepare test sample: img, mask, insts
    test_img = Image.open('../../test_images/ycy.jpg')
    mask = test_img.getchannel(0).point(lambda i: i < 200 and 255)
    # mask.show(title='seg_before')
    mask = np.array(mask)
    mask1 = test_img.getchannel(0).point(lambda i: i < 100 and 255)
    # mask1.show(title='inst1_before')
    mask1 = np.array(mask1)
    mask2 = test_img.getchannel(0).point(lambda i: 100 <= i < 200 and 255)
    # mask2.show(title='inst2_before')
    mask2 = np.array(mask2)
    insts = [mask1, mask2]
    sample = {
        'image': test_img,
        'seg_label': mask,
        'inst_label': insts
    }
    print(test_img.size)
    print(mask.shape)
    print(insts[0].shape)

    after_sample = transform_test(sample)
    after_sample['image'].show(title='img_after')
    Image.fromarray(after_sample['seg_label'], mode='L').point(lambda i: i < 1 and 255).show(title='seg_after')
    Image.fromarray(after_sample['inst_label'][0], mode='L').point(lambda i: i < 1 and 255).show(title='inst_after')
    Image.fromarray(after_sample['inst_label'][1], mode='L').point(lambda i: i < 1 and 255).show(title='inst_after')
