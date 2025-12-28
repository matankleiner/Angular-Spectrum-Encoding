# Angular-Spectrum-Encoding

## System Requirements 

### Hardware Requirements 

This code has been tested on a Linux machine (Ubuntu 22.04.4 LTS) with NVIDIA GeForce GTX 2080 Ti GPU. 

GPU is not mandatory, however it expadite training. 

### Software Requirements 

This code has the following dependencies: 
```
  python >= 3.8.12 
  torch >= 1.12.1
  torchvision >= 0.13.1
  numpy >= 1.23.4
  tqdm >= 4.64.0
```
#### Setting an environment 

Create a python virtual environment, install all dependecies using the `requirements.txt` file and then run the code on your computer. 

```
cd DIR_NAME
python3 -m venv VENV_NAME
source VENV_NAME/bin/activate
pip install -r requirements.txt 
```
Installation time should take around 10 minutes. 


## Usage Instructions  

After installation one can run our code. 

### Data

The data used in our work is the MNIST dataset, available via `torchvision`. See `get_data.py`. 

### Hyperparameters

`config.py` include all the hyperparameters used. 

The different hyperparameters used for running different experiemnts are detailed in the paper. 

### Usage

A usage example can be found in `run_trials.py`. 
