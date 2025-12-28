import torch
import torchvision.datasets as datasets


def DataMNISTCalcInit(transforms_train, batch_size, train=True):
    dataset = datasets.MNIST(root='../data', train=train, download=True, transform=transforms_train)
    mask = (dataset.targets != 0) & (dataset.targets != 9)
    dataset.data = dataset.data[mask]
    dataset.targets = dataset.targets[mask]
    num_samples = 8000
    indices = torch.randperm(len(dataset))[:num_samples]
    dataset.data = dataset.data[indices]
    dataset.targets = dataset.targets[indices]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True)
    return dataloader
