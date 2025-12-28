import os
import gzip
import torch
import numpy as np

from layers import FreeSpaceProp, Lens, circular
from model import PhaseMaskModel
from utils import rs_tf_kernel, set_seed
from config import PIXEL_SIZE
from get_data import DataMNISTCalcInit

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm


def PSNR(pred, target):
    mse = torch.nn.functional.mse_loss(pred, target)
    psnr = 10 * torch.log10(1 / mse)
    return psnr


def gt_typing(labels, numbers):
    # Get the ground truth typing for the given labels and numbers
    gt = []
    for l in labels:
        gt.append(numbers[l])
    return torch.cat(gt, dim=0)


def labels_add(labels, numbers):
    # Get the ground truth typing for the given labels and numbers
    gt = []
    for l in labels:
        gt.append(numbers[l+1])
    return torch.cat(gt, dim=0)


def labels_subtract(labels, numbers):
    # Get the ground truth typing for the given labels and numbers
    gt = []
    for l in labels:
        gt.append(numbers[l-1])
    return torch.cat(gt, dim=0)


def up_to_10(labels, numbers):
    # Get the ground truth typing for the given labels and numbers
    gt = []
    for l in labels:
        gt.append(numbers[10-l])
    return torch.cat(gt, dim=0)

if __name__ == '__main__':
    set_seed(42)
    gpu_number = 0
    device = torch.device(f"cuda:{gpu_number}" if torch.cuda.is_available() else "cpu")
    print("current device:", device)

    paths = [#'trials/optical_4im2im_kspace_rings_2pm/checkpoints/ckpt1000.pth',
             #'trials/optical_4im2im_kspace_rings_4pm/checkpoints/ckpt1000.pth',
             #'trials/optical_4im2im_kspace_bigger_rings_2pm/checkpoints/ckpt1000.pth',
            #  'trials/optical_4im2im_kspace_bigger_rings_4pm/checkpoints/ckpt1000.pth',
             #'trials/optical_4im2im_kspace_squares_2pm/checkpoints/ckpt1000.pth',
            #  'trials/optical_4im2im_kspace_squares_4pm/checkpoints/ckpt1000.pth',
             #'trials/optical_4im2im_kspace_learned_2pm/checkpoints/ckpt1000.pth',
             'trials/optical_4im2im_kspace_learned_4pm/checkpoints/ckpt1000.pth',
             ] 

    # paths = [#'trials/optical_4im2im_kspace_rings_2pm_norm_1/checkpoints/ckpt1000.pth',
    #         #  'trials/optical_4im2im_kspace_rings_4pm/checkpoints/ckpt10.pth',
    #         #  'trials/optical_4im2im_kspace_bigger_rings_2pm_norm_1/checkpoints/ckpt1000.pth',
    #          'trials/optical_4im2im_kspace_bigger_rings_4pm_illum_025/checkpoints/ckpt1000.pth',
    #          'trials/optical_4im2im_kspace_bigger_rings_4pm_illum_05/checkpoints/ckpt1000.pth',
    #          'trials/optical_4im2im_kspace_bigger_rings_4pm_illum_075/checkpoints/ckpt1000.pth',
    #         #  'trials/optical_4im2im_kspace_bigger_rings_4pm_w_norm/checkpoints/ckpt1000.pth',
    #         #  'trials/optical_4im2im_kspace_learned_4pm_w_norm/checkpoints/ckpt1000.pth',
    #         #  'trials/optical_4im2im_kspace_learned_4pm_no_max_w_norm/checkpoints/ckpt1000.pth',
    #         #  'trials/optical_4im2im_kspace_squares_2pm_norm_1/checkpoints/ckpt1000.pth',
    #         #  'trials/optical_4im2im_kspace_learned_4pm/checkpoints/ckpt1000.pth',
    #          ] 


    for path in paths:
        state = torch.load(path, map_location=device)
        
        transforms_train = transforms.Compose([transforms.ToTensor(),
                                               transforms.Pad(padding=(state['args'].pad,
                                                                       state['args'].pad, 
                                                                       state['args'].pad, 
                                                                       state['args'].pad))])

        test_loader = DataMNISTCalcInit(transforms_train, batch_size=10, train=False)

        numbers = []
        for i in range(10):
            image = Image.open(f'/home/tiras/Matan/SharedFolder/MultiCoherence/OpticalTranslation_kspace_4_uni/numbers/{i}.png')
            image = image.convert('L')
            image = image.resize((state['args'].obj_shape, state['args'].obj_shape))
            image = TF.to_tensor(image).unsqueeze(0).to(device)
            numbers.append(image)

        numbers_mean = 78.8063
        if state['args'].filters_shape=="squares" or state['args'].filters_shape=="learned":
            filters_mean1 = 66176.9453125
            filters_mean2 = 63847.30859375
            filters_mean3 = 60222.57421875
            filters_mean4 = 62476.51953125
        elif state['args'].filters_shape=="rings" and state['args'].radius1==0.08e-2:
            filters_mean1 = 9689.376953125
            filters_mean2 = 6929.7265625
            filters_mean3 = 4948.26806640625
            filters_mean4 = 954.0879516601562
        elif state['args'].filters_shape=="rings" and state['args'].radius1==0.16e-2:
            filters_mean1 = 56297.5390625
            filters_mean2 = 51013.98046875
            filters_mean3 = 29934.8984375
            filters_mean4 = 10111.2626953125
        filters_mean = [filters_mean1, filters_mean2, filters_mean3, filters_mean4]


        # build the model and send it to the device
        SHAPE = state['args'].obj_shape + 2 * state['args'].pad 
           
        models = []
        for i in range(state['args'].pm_number):
            models.append(PhaseMaskModel(state['args'].pm_shape, (SHAPE-state['args'].pm_shape)//2).to(device))
        
        for i, model in enumerate(models):
            model.load_state_dict(state[f'models_{i}'])
            model.eval()


        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        kernels5 = rs_tf_kernel(SHAPE, PIXEL_SIZE, state['args'].wavelength, state['args'].prop_dist5, device)
        kernels20 = rs_tf_kernel(1100, PIXEL_SIZE, state['args'].wavelength, state['args'].prop_dist20, device)

        free_space_prop5 = FreeSpaceProp(kernels5)
        free_space_prop20 = FreeSpaceProp(kernels20)

        lens = Lens(shape=SHAPE, pixel_size=PIXEL_SIZE, wavelength=state['args'].wavelength, focal_distance=state['args'].prop_dist20, device=device)
        if state['args'].filters_shape=="rings":
            filter1 = circular(shape=SHAPE, pixel_size=PIXEL_SIZE, rad1=state['args'].radius1, rad2=state['args'].radius2, device=device)
            filter2 = circular(shape=SHAPE, pixel_size=PIXEL_SIZE, rad1=state['args'].radius2, rad2=state['args'].radius3, device=device)
            filter3 = circular(shape=SHAPE, pixel_size=PIXEL_SIZE, rad1=state['args'].radius3, rad2=state['args'].radius4, device=device)
            filter4 = circular(shape=SHAPE, pixel_size=PIXEL_SIZE, rad1=state['args'].radius4, rad2=0, device=device)
        elif state['args'].filters_shape=="squares":
            filter1 = torch.zeros((1, 1, SHAPE, SHAPE), device=device)
            filter1[:,:,:SHAPE//2,:SHAPE//2] = 1
            filter2 = torch.zeros((1, 1, SHAPE, SHAPE), device=device)
            filter2[:,:,SHAPE//2:,:SHAPE//2:] = 1
            filter3 = torch.zeros((1, 1, SHAPE, SHAPE), device=device)
            filter3[:,:,SHAPE//2:SHAPE,SHAPE//2:SHAPE] = 1
            filter4 = torch.zeros((1, 1, SHAPE, SHAPE), device=device)
            filter4[:,:,:SHAPE//2,SHAPE//2:SHAPE] = 1
        elif state['args'].filters_shape == 'learned':
            filter1 = state['filters'][0]
            filter2 = state['filters'][1]
            filter3 = state['filters'][2]
            filter4 = state['filters'][3]
            filter1 = torch.clamp(filter1, 0, 1)
            filter2 = torch.clamp(filter2, 0, 1)
            filter3 = torch.clamp(filter3, 0, 1)
            filter4 = torch.clamp(filter4, 0, 1)
            # thresholding, all the values above 0.5 go to 1, below go to 0
            filter1 = (filter1 > 0.5).float()
            filter2 = (filter2 > 0.5).float()
            filter3 = (filter3 > 0.5).float()
            filter4 = (filter4 > 0.5).float()

        plane_wave = torch.ones(1, 1, 1100, 1100, device=device) 

        filtered_illumination_space1 = free_space_prop20(lens(free_space_prop20(plane_wave * torch.nn.functional.pad(torch.clamp(filter1, 0, 1), pad=(400,400,400,400))))) 
        filtered_illumination_space2 = free_space_prop20(lens(free_space_prop20(plane_wave * torch.nn.functional.pad(torch.clamp(filter2, 0, 1), pad=(400,400,400,400)))))
        filtered_illumination_space3 = free_space_prop20(lens(free_space_prop20(plane_wave * torch.nn.functional.pad(torch.clamp(filter3, 0, 1), pad=(400,400,400,400))))) 
        filtered_illumination_space4 = free_space_prop20(lens(free_space_prop20(plane_wave * torch.nn.functional.pad(torch.clamp(filter4, 0, 1), pad=(400,400,400,400))))) 

        # filtered_illumination_space1 = filtered_illumination_space1[:,:,400:-400,400:-400] / torch.max(torch.abs(filtered_illumination_space1[:,:,400:-400,400:-400]))
        # filtered_illumination_space2 = filtered_illumination_space2[:,:,400:-400,400:-400] / torch.max(torch.abs(filtered_illumination_space2[:,:,400:-400,400:-400]))
        # filtered_illumination_space3 = filtered_illumination_space3[:,:,400:-400,400:-400] / torch.max(torch.abs(filtered_illumination_space3[:,:,400:-400,400:-400]))
        # filtered_illumination_space4 = filtered_illumination_space4[:,:,400:-400,400:-400] / torch.max(torch.abs(filtered_illumination_space4[:,:,400:-400,400:-400]))


        running_PSNR_sub = 0
        running_ssim_sub = 0
        running_PSNR_add = 0
        running_ssim_add = 0
        running_PSNR_10 = 0
        running_ssim_10 = 0
        running_PSNR_typing = 0
        running_ssim_typing = 0
        running_energy1 = 0
        running_energy2 = 0
        running_energy3 = 0
        running_energy4 = 0

        for data in tqdm(test_loader):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            for i in range(4):   
                if i==0:
                    inputs_illum = inputs * filtered_illumination_space1[:,:,400:-400,400:-400]
                    x = free_space_prop5(inputs_illum)
                    for model in models:
                        x = model(x)
                        x = free_space_prop5(x)
                    intensity = torch.sum((torch.abs(x))**2, dim=1, keepdim=True)
                    # norm = numbers_mean / filters_mean[i]
                    norm = torch.sum(labels_subtract(labels, numbers), dim=(1, 2, 3), keepdim=True) \
                        / torch.sum(intensity[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad], dim=(1, 2, 3), keepdim=True)
                    PSNR_sub = PSNR(norm*intensity[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad], labels_subtract(labels, numbers))
                    SSIM_sub = ssim(norm*intensity[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad], labels_subtract(labels, numbers))
                    # PSNR_sub = PSNR(intensity[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad], labels_subtract(labels, numbers))
                    # SSIM_sub = ssim(intensity[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad], labels_subtract(labels, numbers))
                    energy1 = torch.sum(intensity[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad]) / \
                                torch.sum((torch.abs(inputs_illum)**2)[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad])
                elif i==1:   
                    inputs_illum = inputs * filtered_illumination_space2[:,:,400:-400,400:-400]
                    x = free_space_prop5(inputs_illum)
                    for model in models:
                        x = model(x)
                        x = free_space_prop5(x)
                    intensity = torch.sum((torch.abs(x))**2, dim=1, keepdim=True)
                    # norm = numbers_mean / filters_mean[i]
                    norm = torch.sum(labels_add(labels, numbers), dim=(1, 2, 3), keepdim=True) \
                        / torch.sum(intensity[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad], dim=(1, 2, 3), keepdim=True)
                    PSNR_add = PSNR(norm*intensity[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad], labels_add(labels, numbers))
                    SSIM_add = ssim(norm*intensity[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad], labels_add(labels, numbers))
                    # PSNR_add = PSNR(intensity[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad], labels_add(labels, numbers))
                    # SSIM_add = ssim(intensity[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad], labels_add(labels, numbers))
                    energy2 = torch.sum(intensity[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad]) / \
                                torch.sum((torch.abs(inputs_illum)**2)[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad])
                elif i==2:
                    inputs_illum = inputs * filtered_illumination_space3[:,:,400:-400,400:-400]
                    x = free_space_prop5(inputs_illum)
                    for model in models:
                        x = model(x)
                        x = free_space_prop5(x)
                    intensity = torch.sum((torch.abs(x))**2, dim=1, keepdim=True)
                    # norm = numbers_mean / filters_mean[i]
                    norm = torch.sum(up_to_10(labels, numbers), dim=(1, 2, 3), keepdim=True) \
                        / torch.sum(intensity[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad], dim=(1, 2, 3), keepdim=True)
                    PSNR_10 = PSNR(norm*intensity[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad], up_to_10(labels, numbers))
                    SSIM_10 = ssim(norm*intensity[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad], up_to_10(labels, numbers))
                    # PSNR_10 = PSNR(intensity[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad], up_to_10(labels, numbers))
                    # SSIM_10 = ssim(intensity[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad], up_to_10(labels, numbers))
                    energy3 = torch.sum(intensity[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad]) / \
                                torch.sum((torch.abs(inputs_illum)**2)[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad])
                elif i==3:   
                    inputs_illum = inputs * filtered_illumination_space4[:,:,400:-400,400:-400]
                    x = free_space_prop5(inputs_illum)
                    for model in models:
                        x = model(x)
                        x = free_space_prop5(x)
                    intensity = torch.sum((torch.abs(x))**2, dim=1, keepdim=True)
                    # norm = numbers_mean / filters_mean[i]
                    norm = torch.sum(gt_typing(labels, numbers), dim=(1, 2, 3), keepdim=True) \
                        / torch.sum(intensity[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad], dim=(1, 2, 3), keepdim=True)
                    PSNR_typing = PSNR(norm*intensity[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad], gt_typing(labels, numbers))
                    SSIM_typing = ssim(norm*intensity[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad], gt_typing(labels, numbers))
                    # PSNR_typing = PSNR(intensity[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad], gt_typing(labels, numbers))
                    # SSIM_typing = ssim(intensity[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad], gt_typing(labels, numbers))
                    energy4 = torch.sum(intensity[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad]) / \
                                torch.sum((torch.abs(inputs_illum)**2)[:,:, state['args'].pad:-state['args'].pad, state['args'].pad:-state['args'].pad])

            running_PSNR_sub += PSNR_sub.data.item()
            running_ssim_sub += SSIM_sub.data.item()
            running_PSNR_add += PSNR_add.data.item()
            running_ssim_add += SSIM_add.data.item()
            running_PSNR_10 += PSNR_10.data.item()
            running_ssim_10 += SSIM_10.data.item()
            running_PSNR_typing += PSNR_typing.data.item()
            running_ssim_typing += SSIM_typing.data.item()
            running_energy1 += energy1.data.item()
            running_energy2 += energy2.data.item()
            running_energy3 += energy3.data.item()
            running_energy4 += energy4.data.item()

        running_PSNR_sub /= len(test_loader)
        running_ssim_sub /= len(test_loader)
        running_PSNR_add /= len(test_loader)
        running_ssim_add /= len(test_loader)
        running_PSNR_10 /= len(test_loader)
        running_ssim_10 /= len(test_loader)
        running_PSNR_typing /= len(test_loader)
        running_ssim_typing /= len(test_loader)
        running_energy1 /= len(test_loader)
        running_energy2 /= len(test_loader)
        running_energy3 /= len(test_loader)
        running_energy4 /= len(test_loader)

        print(f"For {path}, "
              f"PSNR Sub: {running_PSNR_sub:.4f}, SSIM Sub: {running_ssim_sub:.4f}, "
              f"PSNR Add: {running_PSNR_add:.4f}, SSIM Add: {running_ssim_add:.4f}, "
              f"PSNR 10: {running_PSNR_10:.4f}, SSIM 10: {running_ssim_10:.4f}, "
              f"PSNR Typing: {running_PSNR_typing:.4f}, SSIM Typing: {running_ssim_typing:.4f}, "
              f"Energy 1: {running_energy1:.4f}, Energy 2: {running_energy2:.4f}, "
              f"Energy 3: {running_energy3:.4f}, Energy 4: {running_energy4:.4f}")
