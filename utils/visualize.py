# -- coding:utf8 --
"""
@author: Zheng Shida
@contact: luckyzsd@163.com
"""
import numpy as np
from PIL import Image


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


def make_palette_64(num_classes):
    """
    create palette according to classes (for n_c > 27)
    :param num_classes: int, num of classes
    :return: palette
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for k in range(0, num_classes):
        palette[k, 0] = k & 3  # 000011
        palette[k, 1] = (k & 12) >> 2  # 001100
        palette[k, 2] = (k & 48) >> 4  # 110000
    palette *= 85
    return palette


def np2pil(numpy_img, n_class=26):
    """
    convert numpy_img to PIL.Image via palette
    :param numpy_img: numpy.array, values are [0,1,2,...,n_class]
    :param n_class: num of classes
    :return: PIL.Image
    """
    numpy_img = numpy_img.astype(np.uint8)
    if n_class == 2:
        numpy_img[numpy_img == 1] = 255
        pil_img = Image.fromarray(numpy_img)
    else:
        pil_img = Image.fromarray(numpy_img).convert('P')
        if n_class < 27:
            voc_palette = make_palette_27(n_class)
        else:
            voc_palette = make_palette_64(n_class)
        palette = np.zeros((256, 3), dtype=np.uint8)
        palette[0:n_class, :] = voc_palette
        palette = palette.reshape(768)
        pil_img.putpalette(palette)
    return pil_img


if __name__ == '__main__':
    np_pic = np.zeros((500, 500)).astype(np.uint8)
    for i in range(len(np_pic)):
        for j in range(len(np_pic[0])):
            if i + j > 100 and i * j < 10000:
                np_pic[i][j] = 1
            elif i - j > 50:
                np_pic[i][j] = 2
            elif i // (j+1) > 10:
                np_pic[i][j] = 3

    pic = np2pil(np_pic, 4)
    pic.show()



