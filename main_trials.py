import argparse
import os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from model import PhaseMaskModel
from utils import set_seed, circular
from get_data import DataMNISTCalcInit
from training import training
from config import get_args, PIXEL_SIZE
from PIL import Image

if __name__ == '__main__':
    set_seed(seed=30)
    parsed = get_args()

    # load available device
    device = torch.device(f"cuda:{parsed.gpu_number}" if torch.cuda.is_available() else "cpu")
    print("current device:", device)

    transforms_train = transforms.Compose([transforms.ToTensor(),
                                           transforms.Pad(padding=(parsed.pad, parsed.pad, parsed.pad, parsed.pad))])

    train_loader = DataMNISTCalcInit(transforms_train, batch_size=parsed.batch_size, train=True)

    numbers = []
    for i in range(10):
        image = Image.open(f'/path/to/numbers/{i}.png')
        image = image.convert('L')
        image = image.resize((parsed.obj_shape, parsed.obj_shape))
        image = TF.to_tensor(image).unsqueeze(0).to(device)
        numbers.append(image)

    # build the model and send it to the device
    SHAPE = parsed.obj_shape + 2 * parsed.pad 

    models = []
    for i in range(parsed.pm_number):
        models.append(PhaseMaskModel(parsed.pm_shape, (SHAPE-parsed.pm_shape)//2).to(device))

    if parsed.filters_shape=="rings":
        filter1 = circular(shape=SHAPE, pixel_size=PIXEL_SIZE, rad1=parsed.radius1, rad2=parsed.radius2, device=device)
        filter2 = circular(shape=SHAPE, pixel_size=PIXEL_SIZE, rad1=parsed.radius2, rad2=parsed.radius3, device=device)
        filter3 = circular(shape=SHAPE, pixel_size=PIXEL_SIZE, rad1=parsed.radius3, rad2=parsed.radius4, device=device)
        filter4 = circular(shape=SHAPE, pixel_size=PIXEL_SIZE, rad1=parsed.radius4, rad2=0, device=device)
    elif parsed.filters_shape=="squares":
        filter1 = torch.zeros((1, 1, SHAPE, SHAPE), device=device)
        filter1[:,:,:SHAPE//2,:SHAPE//2] = 1
        filter2 = torch.zeros((1, 1, SHAPE, SHAPE), device=device)
        filter2[:,:,SHAPE//2:,:SHAPE//2:] = 1
        filter3 = torch.zeros((1, 1, SHAPE, SHAPE), device=device)
        filter3[:,:,SHAPE//2:SHAPE,SHAPE//2:SHAPE] = 1
        filter4 = torch.zeros((1, 1, SHAPE, SHAPE), device=device)
        filter4[:,:,:SHAPE//2,SHAPE//2:SHAPE] = 1
    if parsed.filters_shape == 'learned':
        filter1 = torch.nn.Parameter(torch.ones(1,1,SHAPE,SHAPE, device=device))
        filter2 = torch.nn.Parameter(torch.ones(1,1,SHAPE,SHAPE, device=device))
        filter3 = torch.nn.Parameter(torch.ones(1,1,SHAPE,SHAPE, device=device))
        filter4 = torch.nn.Parameter(torch.ones(1,1,SHAPE,SHAPE, device=device))

    # optimizer
    optimizer = torch.optim.Adam(models[0].parameters(), lr=parsed.lr)
    for i in range(1, len(models)):
        optimizer.add_param_group({'params' : models[i].parameters()})
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    if parsed.filters_shape == 'learned':
        optimizer.add_param_group({'params' : [filter1], 'lr': parsed.lr})
        optimizer.add_param_group({'params' : [filter2], 'lr': parsed.lr})
        optimizer.add_param_group({'params' : [filter3], 'lr': parsed.lr})
        optimizer.add_param_group({'params' : [filter4], 'lr': parsed.lr})

    # continue training from a checkpoint
    if parsed.continue_training:
        checkpoint = torch.load(parsed.ckpt_path)
        for i, model in enumerate(models):
            model.load_state_dict(checkpoint[f'models_{i}'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        parsed.epoch = checkpoint['epoch']

    training(models, parsed, train_loader, [filter1, filter2, filter3, filter4], numbers, optimizer, scheduler, device)
    if not os.path.isdir(f'trials/{parsed.trial_name}'):
        os.mkdir(f'trials/{parsed.trial_name}')
