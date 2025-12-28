import time
import os
import torch
from utils import rs_tf_kernel
from layers import FreeSpaceProp, Lens
from config import PIXEL_SIZE


def typing(labels, numbers):
    gt = []
    for l in labels:
        gt.append(numbers[l])
    return torch.cat(gt, dim=0)


def add_one(labels, numbers):
    gt = []
    for l in labels:
        gt.append(numbers[l+1])
    return torch.cat(gt, dim=0)


def subtract_one(labels, numbers):
    gt = []
    for l in labels:
        gt.append(numbers[l-1])
    return torch.cat(gt, dim=0)


def ten_minus_label(labels, numbers):
    gt = []
    for l in labels:
        gt.append(numbers[10-l])
    return torch.cat(gt, dim=0)


def training(
        models,
        args,
        train_loader,
        filters,
        numbers,
        optimizer,
        scheduler,
        device,
):

    criterion = torch.nn.MSELoss()

    SHAPE = args.obj_shape + 2 * args.pad

    kernels5 = rs_tf_kernel(SHAPE, PIXEL_SIZE, args.wavelength, args.prop_dist5, device)
    kernels20 = rs_tf_kernel(SHAPE + (2 *  args.lens_pad), PIXEL_SIZE, args.wavelength, args.prop_dist20, device)

    free_space_prop5 = FreeSpaceProp(kernels5)
    free_space_prop20 = FreeSpaceProp(kernels20)

    lens = Lens(shape=SHAPE, lens_shape=(SHAPE + (2 *  args.lens_pad)), pixel_size=PIXEL_SIZE, wavelength=args.wavelength, focal_distance=args.prop_dist20, device=device)
    plane_wave = torch.ones(1, 1, SHAPE + (2 *  args.lens_pad), SHAPE + (2 *  args.lens_pad), device=device)

    for epoch in range(1, args.epochs + 1):
        for model in models:
            model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        epoch_time = time.time()

        for data in train_loader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            for i in range(4):
                filtered_illumination_space = free_space_prop20(lens(free_space_prop20(plane_wave * torch.nn.functional.pad(torch.clamp(filters[i], 0), pad=(args.lens_pad,args.lens_pad,args.lens_pad,args.lens_pad))))) 
                inputs_illum = inputs * filtered_illumination_space[:,:,args.lens_pad:-args.lens_pad,args.lens_pad:-args.lens_pad] 
                x = free_space_prop5(inputs_illum)
                for model in models:
                    x = model(x)
                    x = free_space_prop5(x)
                intensity = torch.sum((torch.abs(x))**2, dim=1, keepdim=True)
                if i==0:
                    loss_sub = criterion(intensity[:,:, args.pad:-args.pad, args.pad:-args.pad], subtract_one(labels, numbers))
                elif i==1:   
                    loss_add = criterion(intensity[:,:, args.pad:-args.pad, args.pad:-args.pad], add_one(labels, numbers))
                elif i==2:
                    loss_10 = criterion(intensity[:,:, args.pad:-args.pad, args.pad:-args.pad], ten_minus_label(labels, numbers))
                elif i==3:   
                    loss_typing = criterion(intensity[:,:, args.pad:-args.pad, args.pad:-args.pad], typing(labels, numbers))

            loss = loss_10 + loss_typing + loss_sub + loss_add 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
                
        running_loss += loss.data.item()
        # Normalizing the loss by the total number of train batches
        running_loss /= len(train_loader)
        # scheduler step 
        scheduler.step(running_loss)
        
        epoch_time = time.time() - epoch_time
        print(f"Epoch: {epoch:0>2}/{args.epochs} | Training Loss: {running_loss:.4f} | Epoch Time: {epoch_time:.2f} secs")

        if epoch % 10 == 0 or epoch == args.epochs:
            print('==> Saving model ...')
            state = {'epoch': epoch, 'args': args, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'filters': filters}
            for i in range(len(models)):
                state[f'models_{i}'] = models[i].state_dict()
            os.makedirs(f'trials/{args.trial_name}/checkpoints', exist_ok=True)
            torch.save(state, f'./trials/{args.trial_name}/checkpoints/ckpt{epoch}.pth')
