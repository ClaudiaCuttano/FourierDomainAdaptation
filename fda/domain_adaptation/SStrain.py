from fda.utils import FDA_source_to_target
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
IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )


def train_UDA_DeepLab_MobileNet_SSL(model, mobile_model, sourceloader, targetloader, pseudotrgloader,  optimizer,  cfg):
    mobile_optimizer = optim.SGD(mobile_model.optim_parameters(cfg), lr=cfg.TRAIN.LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    sourceloader_iter, targetloader_iter = iter(sourceloader), iter(targetloader)
    pseudoloader_iter = iter(pseudotrgloader)

    cudnn.enabled = True
    cudnn.benchmark = True

    model.train()
    model.to(cfg.DEVICE)
    mobile_model.train()
    mobile_model.to(cfg.DEVICE)

    mean_img = torch.zeros(1, 1)
    CEloss = CrossEntropy2d()
    if os.path.exists(Path(cfg.TRAIN.SNAPSHOT_DIR)/ f'output.txt'):
        os.remove(Path(cfg.TRAIN.SNAPSHOT_DIR)/ f'output.txt')

    start_iter = cfg.TRAIN.START_ITER
    for i in range(start_iter, cfg.TRAIN.NUM_STEPS):
        adjust_learning_rate(optimizer, i, cfg.TRAIN.NUM_STEPS, cfg)
        optimizer.zero_grad()
        adjust_learning_rate(mobile_optimizer, i, cfg.TRAIN.NUM_STEPS, cfg)
        mobile_optimizer.zero_grad()                                                      # zero grad

        src_img, src_lbl, _, _ = sourceloader_iter.next()                            # new batch source
        trg_img, trg_lbl, _, _ = targetloader_iter.next()                            # new batch target
        psu_img, psu_lbl, _, _ = pseudoloader_iter.next()

        if mean_img.shape[-1] < 2:
            B, C, H, W = src_img.shape
            mean_img = IMG_MEAN.repeat(B, 1, H, W)

        #-------------------------------------------------------------------#

        # 1. source to target, target to target
        src_in_trg = FDA_source_to_target(src_img, trg_img, L=cfg.TRAIN.LB).real           # src_lbl
        trg_in_trg = trg_img

        # 2. subtract mean
        src_img = src_in_trg.clone() - mean_img                                 # src_1, trg_1, src_lbl
        trg_img = trg_in_trg.clone() - mean_img                                 # trg_1, trg_0, trg_lbl
        psu_img = psu_img.clone()    - mean_img

        #-------------------------------------------------------------------#

        # evaluate and update params #####
        src_img, src_lbl = src_img.to(cfg.DEVICE), src_lbl.long().to(cfg.DEVICE)  # to gpu
        trg_img, trg_lbl = trg_img.to(cfg.DEVICE), trg_lbl.to(cfg.DEVICE)  # to gpu
        psu_img, psu_lbl = psu_img.to(cfg.DEVICE), psu_lbl.long().to(cfg.DEVICE)  # to gpu

        _, _, h, w = src_img.size()
        interp_source = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)

        # TRAIN DEEPLAB
        out_src = model(src_img)
        out_src = interp_source(out_src)
        loss_seg_src = CEloss(out_src, src_lbl.clone(), weight=None)

        out_psu = model(psu_img)
        out_psu = interp_source(out_psu)
        loss_seg_psu = CEloss(out_psu, psu_lbl.clone(), weight=None)

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

        loss_all = loss_seg_src + loss_seg_psu + triger_ent * cfg.TRAIN.ENTW * loss_ent_trg  # loss of seg on src, and ent on s and t
        loss_all.backward()
        optimizer.step()

        # TRAIN MOBILENET
        out_psu_mobile = mobile_model(psu_img)
        out_psu_mobile = interp_source(out_psu_mobile)
        loss_seg_psu_mobile = CEloss(out_psu_mobile, psu_lbl.clone(), weight=None)

        loss_all_mobile = loss_seg_psu_mobile
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
            eval_single(cfg, model, True, file1)
            file1.close()
            mobile_model.train()
            model.train()
            if i >= cfg.TRAIN.EARLY_STOP - 1:
                break
        elif i % 20 == 0:
            print("Iteration: ", i)
        sys.stdout.flush()



