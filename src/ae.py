import torch

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


def test_model(loader_obj, model_, linear=True):
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
    model_obj,
    optimizer_obj,
    loss_obj,
    loader_obj,
    batch_s,
    epoch_num=1,
    model_linear=True,
):
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


def add_noise(img_, noise_int: float):
    noise =  np.random.normal(loc=0, scale=1, size=img_.shape)
    
    #noise overlaid over image
    noisy = np.clip((img_.numpy() + noise*noise_int),0,1)
    noisy_tensor = torch.tensor(noisy, dtype=torch.float).reshape(1, 1, 28, 28)

    return noisy_tensor


def noisy_test(loader_obj, model_: nn.Sequential, linear: bool = True, noise_intensity: float = 0.2):
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
    plt.title("Model's Regenerated Picture")
    plt.show()

    return
