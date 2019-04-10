# -- coding:utf-8 --
"""
@author: Zheng Shida
@contact: luckyzsd@163.com
"""

from easydict import EasyDict as edict
import time

C = edict()
config = C

C.seed = 131024

"""
Common Configs
"""
C.ROOT_DIR = '/data/code/tmp_proj/'  # project root dir, ending with REPO_NAME
C.REPO_NAME = C.ROOT_DIR.split('/')[-2]     # project name

C.LOG_DIR = C.ROOT_DIR + 'logs/'

current_time = time.strftime('%m-%d-%H-%M', time.localtime())
C.LOG_FILE = C.LOG_DIR + 'log_' + current_time + '.log'

"""
For Model
"""
C.PRETRAINED_NAME = 'ImageNet'  # or some others
C.PRETRAINED_PATH = C.ROOT_DIR + 'pretrained_model/resnet501.pth'  # TODO: add resnet501 into pretrained_model
C.MODEL_NAME = 'BBE'  # for switching main networks with similar framework codes

"""
Hyper parameter
"""
C.INIT_LR = 0.02
C.POWER = 0.9
C.MOMENTUM = 0.9
C.WEIGHT_DECAY = 0.0001
C.MAX_ITER = 200000
C.TRAIN_BATCH_SIZE = 24
C.VAL_BATCH_SIZE = 1

"""
Dataset
"""
C.DATASET_NAME = 'coco'
if C.DATASET_NAME == 'coco':
    C.DATA_DIR = '/data/dataset/coco/'    # dataset dir TODO: maybe split it into train/val dirs
    C.NUM_CLASS = 2
    C.NUM_IN_CHANNELS = 3
elif C.DATASET_NAME == 'HikHandArm':
    C.DATA_DIR = '/data/project_data/'
    C.NUM_CLASS = 3
    C.NUM_IN_CHANNELS = 3
else:
    raise NotImplementedError


"""
Training
"""
C.IS_TEST = False
C.OPTIM_METHOD = 'SGD'
C.IS_LR_SEPERATE = True
if C.IS_LR_SEPERATE:  # now implement 3-level lr strategy: 1. backbone, 2. non backbone, 3. bias
    C.LR_INIT_1 = C.INIT_LR
    C.WEIGHT_DECAY_1 = C.WEIGHT_DECAY
    C.LR_INIT_2 = C.INIT_LR * 2
    C.WEIGHT_DECAY_2 = C.WEIGHT_DECAY
    C.LR_INIT_3 = C.INIT_LR * 5
    C.WEIGHT_DECAY_3 = 0
C.LR_CURVE = 'warm-up and poly decay'
C.SAVE_DIR = C.ROOT_DIR + 'saved_model/'
C.SAVE_EPOCH = 1
C.VAL_EPOCH = 1

"""
Loss
"""
C.WEIGHT_SEMA_LOSS = 3
C.WEIGHT_INST_LOSS = 1

"""
Dataloader
"""
C.IS_SHUFFLE = True
C.NUM_WORKER = 8
C.IS_DROP_LAST = True


"""
Tensorboard
"""
C.TENSORBOARD_DIR = C.LOG_DIR + 'tensorboard/'
C.TENSORBOARD_FILE = C.TENSORBOARD_DIR + ''
"""
Others
"""
C.MIN_VALUE = 0.000001


def show_configs(order=False):
    """
    Show all configs for debugging or training.
    """
    if order:  # get displayed configs ordered
        raise NotImplementedError
    else:
        for cfg in C:
            print(str(cfg) + ': ' + str(C[cfg]))


if __name__ == '__main__':
    show_configs()
