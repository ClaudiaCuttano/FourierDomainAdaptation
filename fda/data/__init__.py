import numpy as np
from torch.utils import data
from fda.data.gta5_dataset import GTA5DataSet
from fda.data.cityscapes_dataset import cityscapesDataSet
from fda.data.cityscapes_dataset_label import cityscapesDataSetLabel
from fda.data.cityscapes_dataset_SSL import cityscapesDataSetSSL
from fda.data.synthia_dataset import SYNDataSet

IMG_MEAN = np.array((0.0, 0.0, 0.0), dtype=np.float32)
image_sizes = {'cityscapes': (1024,512), 'gta5': (1280, 720), 'synthia': (1280, 760)}


def CreateSrcDataLoader(cfg):
    if cfg.SOURCE == 'gta5':
        source_dataset = GTA5DataSet(cfg.DATA_SRC_DIRECTORY, cfg.DATA_LIST_SOURCE, crop_size=image_sizes['cityscapes'],
                                      resize=image_sizes['gta5'], mean=IMG_MEAN,
                                      max_iters=cfg.TRAIN.NUM_STEPS * cfg.TRAIN.BATCH_SIZE )
    elif cfg.SOURCE == 'synthia':
        source_dataset = SYNDataSet(cfg.DATA_SRC_DIRECTORY, cfg.DATA_LIST_SOURCE, crop_size=image_sizes['cityscapes'],
                                      resize=image_sizes['synthia'], mean=IMG_MEAN,
                                      max_iters=cfg.TRAIN.NUM_STEPS * cfg.TRAIN.BATCH_SIZE )
    else:
        raise ValueError('The source dataset mush be either gta5 or synthia')
    
    source_dataloader = data.DataLoader( source_dataset, 
                                         batch_size=cfg.TRAIN.BATCH_SIZE,
                                         shuffle=True, 
                                         num_workers=cfg.NUM_WORKERS,
                                         pin_memory=True )    
    return source_dataloader


def CreateTrgDataLoader(cfg, set='train'):
    if set == 'train' or set == 'trainval':
        target_dataset = cityscapesDataSetLabel( cfg.DATA_TGT_DIRECTORY,
                                                 cfg.DATA_LIST_TARGET,
                                                 crop_size=image_sizes['cityscapes'], 
                                                 mean=IMG_MEAN, 
                                                 max_iters=cfg.TRAIN.NUM_STEPS * cfg.TRAIN.BATCH_SIZE,
                                                 set=set )
    elif set == 'val':
        target_dataset = cityscapesDataSetLabel(cfg.DATA_TGT_DIRECTORY,
                                                 cfg.DATA_LIST_TARGET_VAL,
                                                 crop_size=image_sizes['cityscapes'],
                                                 mean=IMG_MEAN,
                                                 set=set)

    if set  == 'train' or set  == 'trainval':
        target_dataloader = data.DataLoader( target_dataset,
                                             batch_size=cfg.TRAIN.BATCH_SIZE,
                                             shuffle=True,
                                             num_workers=cfg.NUM_WORKERS,
                                             pin_memory=True )
    else:
        target_dataloader = data.DataLoader( target_dataset,
                                             batch_size=1, 
                                             shuffle=False, 
                                             pin_memory=True )

    return target_dataloader


def CreateTrgDataSSLLoader(cfg):
    target_dataset = cityscapesDataSet(cfg.DATA_TGT_DIRECTORY,
                                        cfg.DATA_LIST_TARGET,
                                        crop_size=image_sizes['cityscapes'],
                                        mean=IMG_MEAN, 
                                        set="train")
    target_dataloader = data.DataLoader( target_dataset, 
                                         batch_size=1, 
                                         shuffle=False, 
                                         pin_memory=True )
    return target_dataloader



def CreatePseudoTrgLoader(cfg):
    target_dataset = cityscapesDataSetSSL( cfg.DATA_TGT_DIRECTORY,
                                           cfg.DATA_LIST_TARGET,
                                           crop_size=image_sizes['cityscapes'],
                                           mean=IMG_MEAN,
                                           max_iters= cfg.TRAIN.NUM_STEPS * cfg.TRAIN.BATCH_SIZE,
                                           set="train",
                                           label_folder='../../experiments/pseudolabels')
                                           #label_folder='/data/datasets/tmp_DM_gta_fda')

    target_dataloader = data.DataLoader( target_dataset,
                                         batch_size=cfg.TRAIN.BATCH_SIZE,
                                         shuffle=True,
                                         num_workers=cfg.NUM_WORKERS,
                                         pin_memory=True)

    return target_dataloader

