import os

gpu = 0

for filters_type in ['rings', 'squares', 'learned']:
	os.system(f'python main_trials.py --gpu_number 0 --obj_shape 28 --lr 1e-2 --epochs 1000 --pm_number 4 --batch_size 512 --filters_shape {filters_type} ' 
	          f'--trial_name=optical_4im2im_{filters_type}_4pm')


