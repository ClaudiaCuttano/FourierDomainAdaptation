from fda.utils import FDA_source_to_target
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
from pathlib import Path
import numpy as np
import sys
import os
from fda.domain_adaptation.eval_UDA import eval_single
from fda.utils.func import adjust_learning_rate
from fda.utils.loss import CrossEntropy2d
from torch import nn


IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1))


def train_UDA_DeepLab_MobileNet(model, mobile_model, sourceloader, targetloader, optimizer,  cfg):

    mobile_optimizer = optim.SGD(mobile_model.optim_parameters(cfg), lr=cfg.TRAIN.LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    sourceloader_iter, targetloader_iter = iter(sourceloader), iter(targetloader)

    start_iter = cfg.TRAIN.START_ITER
    cudnn.enabled = True
    cudnn.benchmark = True

    model.train()
    mobile_model.train()
    model.to(cfg.DEVICE)
    mobile_model.to(cfg.DEVICE)

    mean_img = torch.zeros(1, 1)
    CEloss = CrossEntropy2d()
    if os.path.exists(Path(cfg.TRAIN.SNAPSHOT_DIR)/ f'output.txt') and cfg.RESTORE is False:
        os.remove(Path(cfg.TRAIN.SNAPSHOT_DIR)/ f'output.txt')

    for i in range(start_iter, cfg.TRAIN.NUM_STEPS):
        adjust_learning_rate(optimizer, i, cfg.TRAIN.NUM_STEPS, cfg)
        optimizer.zero_grad()
        adjust_learning_rate(mobile_optimizer, i, cfg.TRAIN.NUM_STEPS, cfg)
        mobile_optimizer.zero_grad()

        src_img, src_lbl, _, _ = sourceloader_iter.next()  # new batch source
        trg_img, trg_lbl, _, _ = targetloader_iter.next()  # new batch target

        if mean_img.shape[-1] < 2:
            B, C, H, W = src_img.shape
            mean_img = IMG_MEAN.repeat(B, 1, H, W)

        # 1. source to target, target to target
        src_in_trg = FDA_source_to_target(src_img, trg_img, L=cfg.TRAIN.LB).real # src_lbl
        trg_in_trg = trg_img

        src_img = src_in_trg.clone() - mean_img  # src, src_lbl
        trg_img = trg_in_trg.clone() - mean_img  # trg, trg_lbl

        src_img, src_lbl = src_img.to(cfg.DEVICE), src_lbl.long().to(cfg.DEVICE)  # to gpu
        trg_img, trg_lbl = trg_img.to(cfg.DEVICE), trg_lbl.to(cfg.DEVICE)  # to gpu

        _, _, h, w = src_img.size()
        interp_source = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)
        ############################################### TRAIN DEEPLAB
        out_src = model(src_img)
        x = interp_source(out_src)
        loss_seg_src = CEloss(x, src_lbl, weight=None)

        out_tgt = model(trg_img)
        P = torch.softmax(out_tgt, dim=1)  # [B, 19, H, W]
        logP = torch.log_softmax(out_tgt, dim=1)  # [B, 19, H, W]
        PlogP = P * logP  # [B, 19, H, W]
        ent = -1.0 * PlogP.sum(dim=1)  # [B, 1, H, W]
        ent = ent / 2.9444  # change when classes is not 19
        # compute robust entropy
        ent = ent ** 2.0 + 1e-8
        ent = ent ** cfg.TRAIN.ITA
        loss_ent_trg = ent.mean()

        triger_ent = 0.0
        if i > cfg.TRAIN.SWITCHTOENTROPY:
            triger_ent = 1.0

        loss_all = loss_seg_src + triger_ent * cfg.TRAIN.ENTW * loss_ent_trg
        loss_all.backward()
        optimizer.step()

        ########################################## TRAIN MOBILENET
        out_src_mobile = mobile_model(src_img)
        x_mobile = interp_source(out_src_mobile)
        loss_seg_src_mobile = CEloss(x_mobile, src_lbl, weight=None)

        out_tgt_mobile = mobile_model(trg_img)  # forward pass
        P = torch.softmax(out_tgt_mobile, dim=1)  # [B, 19, H, W]
        logP = torch.log_softmax(out_tgt_mobile, dim=1)  # [B, 19, H, W]
        PlogP = P * logP  # [B, 19, H, W]
        ent_mobile = -1.0 * PlogP.sum(dim=1)  # [B, 1, H, W]
        ent_mobile = ent_mobile / 2.9444  # change when classes is not 19
        # compute robust entropy
        ent_mobile = ent_mobile ** 2.0 + 1e-8
        ent_mobile = ent_mobile ** cfg.TRAIN.ITA
        loss_ent_trg_mobile = ent_mobile.mean()

        loss_all_mobile = loss_seg_src_mobile + triger_ent * cfg.TRAIN.ENTW * loss_ent_trg_mobile
        loss_all_mobile.backward()
        mobile_optimizer.step()

        if i % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i != 0:
            print('taking snapshot ...')
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(mobile_model.state_dict(), snapshot_dir / f'mobile_model_{i}_{cfg.TRAIN.LB}.pth')
            torch.save(model.state_dict(), snapshot_dir / f'model_{i}_{cfg.TRAIN.LB}.pth')

        if i % cfg.TRAIN.VALIDATE == 0 and i != 0:
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            file1 = open(snapshot_dir / f'output.txt', "a")
            file1.write("\n ITERATION : " + str(i) + "\n")
            eval_single(cfg, mobile_model, True, file1)
            file1.close()
            mobile_model.train()
            if i >= cfg.TRAIN.EARLY_STOP - 1:
                break
        elif i%20 == 0:
            print("Iteration: ", i)
        sys.stdout.flush()

