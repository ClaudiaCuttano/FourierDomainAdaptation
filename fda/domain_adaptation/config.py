import os.path as osp
import numpy as np
from easydict import EasyDict
from fda.utils import project_root
from fda.utils.serialization import yaml_load

cfg = EasyDict()

# COMMON CONFIGS
# Number of workers for dataloading
cfg.RESTORE = False
cfg.NUM_WORKERS = 4
cfg.DA_METHOD = 'FDA'
cfg.DEVICE = 0
# List of training images
cfg.SOURCE = 'GTA'
cfg.TARGET = 'Cityscapes'
cfg.SET = 'train'
cfg.DATA_LIST_SOURCE = str('fda/dataset/gta_list/train.txt')

cfg.DATA_LIST_TARGET = str(project_root / 'fda/dataset/cityscapes_list/train.txt')
cfg.DATA_LIST_TARGET_VAL = str(project_root / 'fda/dataset/cityscapes_list/val.txt')
# Directories
# URANIA
#cfg.DATA_SRC_DIRECTORY = str('/data/datasets/GTA5')
#cfg.DATA_TGT_DIRECTORY = str('/data/datasets/Cityscapes')
# KRONOS
#cfg.DATA_SRC_DIRECTORY = str('/home/vandal/datasets/GTA5')
#cfg.DATA_SRC_DIRECTORY = str('/home/vandal/datasets/SYNTHIA/RAND_CITYSCAPES/RGB')
#cfg.DATA_TGT_DIRECTORY = str('/home/vandal/datasets/Cityscapes')
# LEGION
#cfg.DATA_SRC_DIRECTORY = str('/home/ccuttano/data/GTA5')
cfg.DATA_SRC_DIRECTORY = str('/home/ccuttano/data/SYNTHIA/RAND_CITYSCAPES')
cfg.DATA_TGT_DIRECTORY = str('/home/ccuttano/data/Cityscapes')

# Number of object classes
cfg.NUM_CLASSES = 19
# Exp dirs
cfg.EXP_NAME = ''
cfg.EXP_ROOT = project_root / 'experiments'
cfg.EXP_ROOT_SNAPSHOT = osp.join(cfg.EXP_ROOT, 'snapshots')
cfg.EXP_ROOT_LOGS = osp.join(cfg.EXP_ROOT, 'logs')
# CUDA
cfg.PIN_MEMORY = False
# TRAIN CONFIGS
cfg.restore_opt1 = ''
cfg.restore_opt2 = ''
cfg.restore_opt3 = ''
cfg.SAVE = ''
cfg.TRAIN = EasyDict()
cfg.TRAIN.SNAPSHOT_DIR = ''
cfg.TRAIN.BATCH_SIZE = 1
cfg.TRAIN.NUM_STEPS = 150000
cfg.TRAIN.EARLY_STOP = 100000
cfg.TRAIN.MODEL = 'DeepLab'
cfg.TRAIN.LB = 0.01
cfg.TRAIN.ITA = 2
cfg.TRAIN.ENTW = 0.005
cfg.TRAIN.SWITCHTOENTROPY = 50000
cfg.TRAIN.RESTORE_FROM = ''
cfg.TRAIN.LEARNING_RATE = 2.5e-4
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 0.0005
cfg.TRAIN.POWER = 0.9
cfg.TRAIN.SAVE_PRED_EVERY = 50000
cfg.TRAIN.VALIDATE = 5000
cfg.TRAIN.START_ITER =0
# Domain adaptation

cfg.TEST = EasyDict()
cfg.TEST.RESTORE_FROM = ''
cfg.TEST.MULTI = False


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not EasyDict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        # if not b.has_key(k):
        if k not in b:
            raise KeyError(f'{k} is not a valid config key')

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(f'Type mismatch ({type(b[k])} vs. {type(v)}) '
                                 f'for config key: {k}')

        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                _merge_a_into_b(a[k], b[k])
            except Exception:
                print(f'Error under config key: {k}')
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options.
    """
    yaml_cfg = EasyDict(yaml_load(filename))
    _merge_a_into_b(yaml_cfg, cfg)
