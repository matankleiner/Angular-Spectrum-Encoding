import torch
import numpy as np
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rs_tf_kernel(shape, pixel_size, wavelength, dist, device):
    size = shape * pixel_size
    fx = torch.arange((-1 / (2 * (pixel_size))), (1 / (2 * (pixel_size))), (1/(size)))
    Fx, Fy = torch.meshgrid(fx, fx, indexing="ij")
    k = 2 * np.pi / wavelength
    H = torch.exp(1j * k * dist * torch.sqrt(1 - (wavelength * Fx)**2 - (wavelength * Fy)**2))
    H = H * torch.where(Fx**2 + Fy**2 < (1/wavelength)**2, 1, 0)
    return torch.fft.fftshift(H).to(device)


def circular(shape, pixel_size, rad1, rad2, device):
    size = shape * pixel_size
    x = torch.linspace(-size/2, size/2, shape, device=device)
    X, Y = torch.meshgrid(x, x, indexing='ij')
    if rad2 == 0:
        arg = torch.sqrt(X ** 2 + Y ** 2) / rad1
        circ = torch.where(arg < 1, 1, 0)    
        pad = int((shape - circ.shape[0]) / 2) 
        circ_pad = torch.nn.functional.pad(circ, pad=(pad, pad, pad, pad)) 
        return circ_pad.unsqueeze(0).unsqueeze(0)
    else: 
        if rad2 > rad1:
            raise ValueError('rad2 must be smaller than rad1')
        else:
            arg1 = torch.sqrt(X ** 2 + Y ** 2) / rad1
            circ1 = torch.where(arg1 < 1, 1, 0)    
            arg2 = torch.sqrt(X ** 2 + Y ** 2) / rad2
            circ2 = torch.where(arg2 < 1, 1, 0)
            pad = int((shape - circ1.shape[0]) / 2) 
            ring_pad = torch.nn.functional.pad(circ1-circ2, pad=(pad, pad, pad, pad)) 
            return ring_pad.unsqueeze(0).unsqueeze(0)
