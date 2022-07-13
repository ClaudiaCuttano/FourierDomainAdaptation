import sys
sys.path.append('/home/ccuttano/FDA')
from fda.domain_adaptation.config import cfg, cfg_from_file
import argparse
import os
import os.path as osp
import pprint
from fda.utils import project_root
from fda.data import CreateSrcDataLoader, CreateTrgDataLoader, CreatePseudoTrgLoader
from fda.model import CreateModel
from fda.domain_adaptation.train_UDA import train_UDA
from fda.domain_adaptation.train_UDA_DeepLab_MobileNet import train_UDA_DeepLab_MobileNet
import torch
from fda.model.mobileNet import MobileNetV2
from fda.domain_adaptation.SStrain import train_UDA_DeepLab_MobileNet_SSL


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument('--cfg', type=str, default=None, help='optional config file', )
    return parser.parse_args()


def main():
    # LOAD ARGS
    args = get_arguments()
    print('Called with args:')
    print(args)
    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg)
    cfg.DATA_LIST_SOURCE = str(project_root) +str(cfg.DATA_LIST_SOURCE)
    print('Using config:')
    pprint.pprint(cfg)

    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.DA_METHOD}_{cfg.TRAIN.LB}'
    if cfg.TRAIN.SNAPSHOT_DIR == '':
        cfg.TRAIN.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)

    sourceloader, targetloader = CreateSrcDataLoader(cfg), CreateTrgDataLoader(cfg)

    model, optimizer = CreateModel(cfg)
    saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM)
    if 'DeepLab_init.pth' in cfg.TRAIN.RESTORE_FROM and cfg.RESTORE==False:
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        model.load_state_dict(new_params)
    elif cfg.RESTORE is True:
        saved_state_dict = torch.load('')
        model.load_state_dict(saved_state_dict)

    if (cfg.TRAIN.MODEL == 'DeepLab_MobileNet' or cfg.TRAIN.MODEL == 'DeepLab_MobileNet_SSL') and cfg.RESTORE==False:
        mobile_model = MobileNetV2(num_classes=cfg.NUM_CLASSES)
        mobile_model.fix_bn()
        saved_state_dict = torch.load('../../pretrained_models/mobilenet_v2-7ebf99e0.pth')
        new_params = mobile_model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            # i_parts[1]!='18': if truncated version
            if not i_parts[0] == 'classifier':
                new_params[i] = saved_state_dict[i]
        mobile_model.load_state_dict(new_params)
    elif cfg.RESTORE is True:
        mobile_model = MobileNetV2(num_classes=cfg.NUM_CLASSES)
        mobile_model.fix_bn()
        saved_state_dict = torch.load('')
        mobile_model.load_state_dict(saved_state_dict)

    if cfg.TRAIN.MODEL == 'DeepLab':
        train_UDA(model, sourceloader, targetloader, optimizer,  cfg)
    elif cfg.TRAIN.MODEL =='DeepLab_MobileNet':
        train_UDA_DeepLab_MobileNet(model, mobile_model, sourceloader, targetloader, optimizer,  cfg)
    elif cfg.TRAIN.MODEL =='DeepLab_MobileNet_SSL':
        pseudotrgloader = CreatePseudoTrgLoader(cfg)
        train_UDA_DeepLab_MobileNet_SSL(model, mobile_model, sourceloader, targetloader, pseudotrgloader, optimizer,  cfg)


if __name__ == '__main__':
    main()

