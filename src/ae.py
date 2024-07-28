import torch
import torchvision

import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt


def make_model(model_object, lr_rate=0.001, compress_=None):
    """Make all of the needed obects for training.

    Args:
        model_object:
            The class which we want to derive the model from.
        lr_rate:
            elarning rate for the optimizer
        compress_:
            the number of neurons at the heart of autoencoder which defines
            how much we are going to compress the data. We use this with linear
            autoencoder.

    Returns:
        A tuple cotanining the initiated model, optimizer and loss function.
    """
    if not compress_:
        model_ = model_object()
    else:
        model_ = model_object(compress_)
    optimizer_ = torch.optim.Adam(model_.parameters(), lr=lr_rate)
    loss_ = nn.MSELoss()

    return model_, optimizer_, loss_


def test_model(loader_obj, model_, linear=True) -> None:
    """Test the output of the autoencoder model by showing the images.

    Args:
        loader_obj:
            The object of the loader for data batches.
        model_:
            The model which we want to test the output.
        linear:
            If te model is linear or CNN.
    """
    batch_iter = iter(loader_obj)
    batch_images = next(batch_iter)
    tmp_image = batch_images[0][0, 0, :, :]
    plt.imshow(tmp_image)
    plt.title("Original Image")
    plt.show()

    plt.figure()
    if linear:
        model_input = tmp_image.reshape(28 * 28)
    else:
        model_input = tmp_image.reshape(1, 1, 28, 28)
    output = model_(model_input)
    plt.imshow(output.detach().numpy().reshape(28, 28))
    plt.title("Model's Regenerated Picture")
    plt.show()

    return


def train_model(
    model_obj: nn.Module,
    optimizer_obj,
    loss_obj,
    loader_obj,
    batch_s: int,
    epoch_num: int = 1,
    model_linear=True,
) -> nn.Module:
    train_loss = []

    for epoch in range(epoch_num):
        for i, data_ in enumerate(loader_obj, 0):
            batches, targets = data_
            if model_linear:
                batches = batches.reshape([batch_s, 28 * 28])

            # zero the parameter gradients
            optimizer_obj.zero_grad()

            # Find the output of the Nerual Net
            # Forward Pass
            logits = model_obj(batches)

            # Calculate the loss
            loss = loss_obj(logits, batches)

            # Update the neural net and gradients
            # Backward Propagation
            loss.backward()
            optimizer_obj.step()

            # print(f"{loss.item():0.5f}")
            # Append the loss of training
            train_loss.append(loss.item())

    plt.plot(train_loss)
    plt.title("Training loss")
    plt.show()

    return model_obj


def add_noise(img_, noise_int: float) -> torch.Tensor:
    """Add noise to the given image.
    
    Args:
        img_:
            The given image.
        noise_int:
            The intensity of the noise, varies between 0 and 1.
    
    Returns:
        A tensor of the noisy image.
    """
    noise = np.random.normal(loc=0, scale=1, size=img_.shape)

    # noise overlaid over image
    noisy = np.clip((img_.numpy() + noise * noise_int), 0, 1)
    noisy_tensor = torch.tensor(noisy, dtype=torch.float).reshape(1, 1, 28, 28)

    return noisy_tensor


def noisy_test(
    loader_obj, model_: nn.Module, linear: bool = True, noise_intensity: float = 0.2
):
    batch_iter = iter(loader_obj)
    batch_images = next(batch_iter)
    tmp_image = batch_images[0][0, 0, :, :]
    plt.imshow(tmp_image)
    plt.title("Original Image")
    plt.show()

    noisy_img = add_noise(tmp_image, noise_intensity)
    plt.figure()
    plt.imshow(noisy_img.reshape(28, 28).numpy())
    plt.title("Noisy Image")
    plt.show()

    plt.figure()
    if linear:
        model_input = noisy_img.reshape(28 * 28)
    else:
        model_input = noisy_img.reshape(1, 1, 28, 28)
    output = model_(model_input)
    plt.imshow(output.detach().numpy().reshape(28, 28))
    plt.title("Model's Regenerated Image")
    plt.show()

    return


def image_show(img_, img_title: str):
    img_ = torchvision.utils.make_grid(img_)
    npimg = img_.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(img_title)
    plt.show()

    return


def test_cifar(cifar_model, data_loader_):
    # get some random training images
    dataiter = iter(data_loader_)
    images, labels = next(dataiter)

    # show images by cinverting batches to grids
    image_show(images, "Original Image")

    out_batch = cifar_model(images)
    image_show(out_batch, "Model's Regenerated Image")

    return
