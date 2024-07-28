import torch
import torchvision
from torchvision import transforms


def make_dataloader(data_, batch_size: int):
    batch_size = 32

    # Make the DataLoader Object
    train_loader = torch.utils.data.DataLoader(
        data_, batch_size=batch_size, shuffle=True, num_workers=2
    )

    return train_loader

def make_transforms():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    return transform

def load_data_general(data_name: str):
    transform_ = make_transforms()

    if data_name=="mnist":
        data_ = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform_
        )
    elif data_name=="cifar":
        data_ = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_
        )
    
    return data_


def load_mnist_data():
    # Load data
    train_data = load_data_general("mnist")
    
    # Make batches of data
    data_loader = make_dataloader(data_=train_data, batch_size=32)

    return data_loader

def load_cifar_data():
    # Load data
    train_data = load_data_general("cifar")

    # Make batches of data
    data_loader = make_dataloader(data_=train_data, batch_size=32)
    
    return data_loader
