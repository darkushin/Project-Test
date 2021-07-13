import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.fft


def display_fourier_im(fourier_space_im, title):
    plt.imshow(np.log(1 + np.abs(fourier_space_im)))
    plt.title(title)
    plt.show()


def display_im(im, title):
    plt.imshow(im)
    plt.title(title)
    plt.show()


def double_frames_fourier(imgs):
    """
    Double the number of frames (in time) of the given tensor.
    Input format should be (Batch_Size, Channels, T, H, W)
    """
    b, c, t, h, w = imgs.shape

    # fourier space image:
    torch_imgs = imgs.clone().detach()
    f = torch.fft.fftn(torch_imgs)
    f_shifted = torch.fft.fftshift(f)

    # double number of images in fourier space:
    f_scaled = torch.zeros((b, c, 2*t, h, w), dtype=torch.complex64)
    f_scaled[:, :, t//2: 3*t//2, :, :] = f_shifted

    # shift back:
    f_scaled_shifted = torch.fft.fftshift(f_scaled)

    # spatial domain:
    inv_img = torch.fft.ifftn(f_scaled_shifted)  # inverse F.T.
    filtered_img = torch.abs(inv_img)
    return filtered_img







