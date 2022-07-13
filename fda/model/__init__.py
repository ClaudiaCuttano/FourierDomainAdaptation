import torch.optim as optim
from fda.model.deeplabv2 import Res_Deeplab


def CreateModel(cfg):
    if cfg.TRAIN.MODEL == 'DeepLab' or cfg.TRAIN.MODEL == 'DeepLab_MobileNet' or cfg.TRAIN.MODEL == 'DeepLab_MobileNet_SSL':
        model = Res_Deeplab(num_classes=cfg.NUM_CLASSES)
        optimizer = optim.SGD(model.optim_parameters(cfg),
                            lr=cfg.TRAIN.LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        optimizer.zero_grad()
        return model, optimizer


def CreateSSLModel(cfg):
    model = Res_Deeplab(num_classes=cfg.NUM_CLASSES)
    return model

