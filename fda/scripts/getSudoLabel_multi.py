import sys
sys.path.append('/home/ccuttano/FDA')
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
from fda.data import CreateTrgDataSSLLoader
from tqdm import tqdm
import os
import argparse
from fda.domain_adaptation.config import cfg, cfg_from_file
from fda.model import CreateSSLModel

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

    if not os.path.exists(cfg.SAVE):
        os.makedirs(cfg.SAVE)

    model1 = CreateSSLModel(cfg)
    saved_state_dict = torch.load(cfg.restore_opt1)
    model1.load_state_dict(saved_state_dict)
    model1.eval()
    model1.cuda()

    model2= CreateSSLModel(cfg)
    saved_state_dict = torch.load(cfg.restore_opt2)
    model2.load_state_dict(saved_state_dict)
    model2.eval()
    model2.cuda()

    model3 = CreateSSLModel(cfg)
    saved_state_dict = torch.load(cfg.restore_opt3)
    model3.load_state_dict(saved_state_dict)
    model3.eval()
    model3.cuda()

    targetloader = CreateTrgDataSSLLoader(cfg)

    # change the mean for different dataset
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    IMG_MEAN = torch.reshape(torch.from_numpy(IMG_MEAN), (1,3,1,1))
    mean_img = torch.zeros(1, 1)

    predicted_label = np.zeros((len(targetloader), 512, 1024))
    predicted_prob = np.zeros((len(targetloader), 512, 1024))
    image_name = []
    test_iter = iter(targetloader)
    with torch.no_grad():
        for index in tqdm(range(len(targetloader))):
            image, _, name = next(test_iter)  # 1. get image
            if mean_img.shape[-1] < 2:
                B, C, H, W = image.shape
                mean_img = IMG_MEAN.repeat(B,1,H,W)
            image = image.clone() - mean_img
            image = Variable(image).cuda()

            # forward
            output1 = model1(image)
            output1 = nn.functional.softmax(output1, dim=1)

            output2 = model2(image)
            output2 = nn.functional.softmax(output2, dim=1)

            output3 = model3(image)
            output3 = nn.functional.softmax(output3, dim=1)

            a, b = 0.3333, 0.3333
            output = a*output1 + b*output2 + (1.0-a-b)*output3

            output = nn.functional.interpolate(output, (512, 1024), mode='bilinear', align_corners=True).cpu().data[0].numpy()
            output = output.transpose(1,2,0)
       
            label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
            predicted_label[index] = label.copy()
            predicted_prob[index] = prob.copy()
            image_name.append(name[0])
        
    thres = []
    for i in range(19):
        x = predicted_prob[predicted_label==i]
        if len(x) == 0:
            thres.append(0)
            continue        
        x = np.sort(x)
        thres.append(x[int(np.round(len(x)*0.66))])
    print( thres )
    thres = np.array(thres)
    thres[thres>0.9]=0.9
    print(thres)

    for index in range(len(targetloader)):
        name = image_name[index]
        label = predicted_label[index]
        prob = predicted_prob[index]
        for i in range(19):
            label[(prob < thres[i]) * (label == i)] = 255
        output = np.asarray(label, dtype=np.uint8)
        output = Image.fromarray(output)
        name = name.split('/')[-1]
        output.save('%s/%s' % (cfg.SAVE, name))

    
if __name__ == '__main__':
    main()
    
