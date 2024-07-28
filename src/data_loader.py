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


def load_mnist_data():
    # Define the transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # Load data
    train_data = torchvision.datasets.FashionMNIST(
        root="data", train=True, download=True, transform=transform
    )

    data_loader = make_dataloader(data_=train_data, batch_size=32)

    return data_loader
