import argparse

SAMPLES = 5000 
PIXEL_SIZE = 10e-6 


def get_args():
    parser = argparse.ArgumentParser()
    
    # optimization hyperparameters 
    parser.add_argument('--batch_size', help='batch size', default=512, type=int)
    parser.add_argument('--lr', help='learning rate', default=1e-2, type=float)
    parser.add_argument('--epochs', help='number of epochs', default=1000, type=int)
    
    # optical properties 
    parser.add_argument('--prop_dist5', help='propagation distance between layers, in meter', default=5e-2, type=float)
    parser.add_argument('--prop_dist20', help='propagation distance between layers, in meter', default=20e-2, type=float)
    parser.add_argument('--wavelength', help='', default=550e-9, type=float)
    parser.add_argument('--radius1', help='radius of the first circular aperture, in meter', default=0.16e-2, type=float)
    parser.add_argument('--radius2', help='radius of the second circular aperture, in meter', default=0.12e-2, type=float)
    parser.add_argument('--radius3', help='radius of the second circular aperture, in meter', default=0.08e-2, type=float)
    parser.add_argument('--radius4', help='radius of the second circular aperture, in meter', default=0.04e-2, type=float)
    parser.add_argument('--filters_shape', help='shape of the filters', default='[rings, squares, learned]', type=str)

    # trial properties
    parser.add_argument('--obj_shape', help='object shape', default=28, type=int)
    parser.add_argument('--pm_shape', help='pm shape', default=300, type=int)
    parser.add_argument('--pad', help='padding', default=136, type=int)
    parser.add_argument('--lens_pad', help='padding', default=400, type=int)
    parser.add_argument('--pm_number', help='phase mask number', default=4, type=int)

    # dataset to use 
    parser.add_argument('--dataset', help='dataset to use', default='MNIST', type=str)
    
    # trial configurations      
    parser.add_argument('--trial_name', help='the trial name', default='trial', type=str)
    parser.add_argument('--gpu_number', help='gpu to use', default=0, type=int) 

    # continue training 
    parser.add_argument('--continue_training', help='continue training from a checkpoint', action='store_true')
    parser.add_argument('--ckpt_path', help='path to checkpoint', type=str)
    
    return parser.parse_args()
