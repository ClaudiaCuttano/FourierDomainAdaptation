import torch
import numpy as np
import pathlib

project_root = pathlib.Path(__file__).resolve().parents[2]
__all__ = ['project_root']


def low_freq_mutate( amp_src, amp_trg, L=0.1 ):
    _, _, h, w = amp_src.size()
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)     # get b
    amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      # top left
    amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # top right
    amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # bottom left
    amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # bottom right
    return amp_src


def FDA_source_to_target(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    # get fft of both source and target
    fft_src = torch.fft.fftn(src_img.clone(), dim=(2, 3)) # check if fft2 is enough
    fft_trg = torch.fft.fftn(trg_img.clone(), dim=(2, 3))

    # extract amplitude and phase of both ffts
    amp_src, pha_src = fft_src.abs(), fft_src.angle()
    amp_trg, pha_trg = fft_trg.abs(), fft_trg.angle()

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=L )

    # recompose fft of source
    # recompose fft of source
    fft_src_real = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_imag = torch.sin(pha_src.clone()) * amp_src_.clone()
    fft_src_ = torch.complex(fft_src_real, fft_src_imag)

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.fft.ifftn(fft_src_, dim=(2, 3))

    return src_in_trg


