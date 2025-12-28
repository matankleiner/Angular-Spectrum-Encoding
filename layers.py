from networkx import radius
import torch
import torch.nn as nn
from numpy import pi


class FreeSpaceProp(nn.Module):
    # The angular spectrum method with Rayleigh–Sommerfeld diffraction formulation
    def __init__(self, kernel_fft):
        super().__init__()
        self.kernel_fft = kernel_fft

    def forward(self, x):
        x_fft = torch.fft.fft2(torch.fft.fftshift(x, dim=(2,3)))
        out = torch.fft.ifftshift(torch.fft.ifft2(x_fft * self.kernel_fft), dim=(2,3))
        return out


class PhaseMask(nn.Module):

    def __init__(self, shape, pad):

        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(1, 1, shape, shape))
        self.pad = pad

    def forward(self, x):
        pm = torch.nn.functional.pad(torch.exp(1j * self.weights), (self.pad, self.pad, self.pad, self.pad))
        return x * pm
    

class Lens(nn.Module):

    def __init__(self, shape, lens_shape, pixel_size, wavelength, focal_distance, device):

        super().__init__()
        size = shape * pixel_size
        x = torch.linspace(-size/2, size/2, shape)
        X, Y = torch.meshgrid(x, x, indexing='ij')
        k = 2 * pi / wavelength
        self.pad = (lens_shape - shape) // 2
        self.phase = torch.exp(-1j * (k / (2 * focal_distance)) * (X**2 + Y**2)).to(device)

    def forward(self, x):
        return x * torch.nn.functional.pad(self.phase.unsqueeze(0).unsqueeze(0), (self.pad, self.pad, self.pad, self.pad))
    