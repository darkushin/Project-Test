import numpy as np
import matplotlib.pyplot as plt
import torch


def display_fourier_im(fourier_space_im, title):
    plt.imshow(np.log(1 + np.abs(fourier_space_im)))
    plt.title(title)
    plt.show()


def display_im(im, title):
    im -= im.min()
    im = im * 255 / im.max()
    im = im.astype(np.uint8)
    plt.imshow(im)
    plt.title(title)
    plt.show()


def double_frames_fourier(imgs):
    """
    Double the number of frames (in time) of the given tensor.
    Input format should be (Batch_Size, Channels, T, H, W)
    """
    numpy_imgs = imgs.detach().cpu().numpy()
    b, c, t, h, w = imgs.shape

    # fourier space image:
    f = np.fft.fftn(numpy_imgs)
    f_shifted = np.fft.fftshift(f)

    # double number of images in fourier space:
    f_scaled = np.zeros((b, c, 2*t, h, w), dtype=np.complex64)
    f_scaled[:, :, t//2: 3*t//2, :, :] = f_shifted

    # shift back:
    f_scaled_shifted = np.fft.fftshift(f_scaled)

    # spatial domain:
    inv_img = np.fft.ifftn(f_scaled_shifted)  # inverse F.T.
    filtered_img = np.abs(inv_img)
    torch_imgs = torch.from_numpy(filtered_img)
    return torch_imgs

