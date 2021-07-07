import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
from skimage.color import rgb2gray
import torch_resizer
import torch
import torch.fft


def im_fft(im):
    """
    Perform FFT on the given input image.
    """
    fft_im = np.fft.fft2(im)
    return fft_im


def im_ifft(fft_im):
    """
    Perform IFFT on the given input image.
    """
    ifft_im = np.fft.ifft2(fft_im)
    return ifft_im


def pad_im(im):
    """
    Pad the given image by inserting 0 every second line and raw.
    """
    padded_im = np.zeros((2*im.shape[0], 2*im.shape[1]))
    padded_im[padded_im.shape[0]//4:padded_im.shape[0]//4+im.shape[0], padded_im.shape[1]//4:padded_im.shape[
                                                                                              1]//4+im.shape[1]] = im

    return padded_im


def read_image(filename):
    """
    Read the given image.
    """
    im = imageio.imread(filename).astype(np.float64) / 255
    im = rgb2gray(im)
    return im


def filter_image(im):
    filter = np.zeros(im.shape)
    filter[filter.shape[0]//2 - 50:filter.shape[0]//2 + 50, filter.shape[1]//2-50:filter.shape[1]//2+50] = 1
    filtered_im = np.multiply(im, filter)
    return filtered_im


def upscale_im(img):
    # original image:
    # plt.imshow(img)
    # plt.title('orig')
    plt.imsave('orig.png', img)
    h, w, c = img.shape

    # fourier space image:
    f = np.fft.fftn(img.astype(np.float32))
    f_shifted = np.fft.fftshift(f)
    # plt.imshow(np.log(1+np.abs(f_shifted)))
    # plt.title('fourier space')
    # plt.show()

    # upscaled in fourier space:
    f_filtered = np.zeros((2*h, 2*w, c)).astype(np.complex64)
    f_filtered[h//2:3*h//2, w//2:3*w//2, :] = f_shifted
    # plt.imshow(np.log(1 + np.abs(f_filtered)))
    # plt.title('scaled fourier space')
    # plt.show()

    # shift back:
    f_filtered_shifted = np.fft.fftshift(f_filtered)
    # plt.imshow(np.log(1 + np.abs(f_filtered_shifted)))
    # plt.title('shifted scaled fourier space')
    # plt.show()

    # spatial domain:
    inv_img = np.fft.ifftn(f_filtered_shifted)  # inverse F.T.
    filtered_img = np.abs(inv_img)
    filtered_img -= filtered_img.min()
    filtered_img = filtered_img * 255 / filtered_img.max()
    filtered_img = filtered_img.astype(np.uint8)
    # plt.imshow(filtered_img)
    # plt.title('upscaled')
    # plt.show()
    plt.imsave('upscaled.png', filtered_img)


def double_frames_fourier(imgs):
    # original image:
    # plt.imshow(img)
    # plt.title('orig')
    # plt.imsave('orig.png', img)
    t, h, w, c = imgs.shape

    # fourier space image:
    f = torch.fft.fftn(torch.tensor(imgs, dtype=torch.float32))
    f_shifted = torch.fft.fftshift(f)
    # plt.imshow(np.log(1+np.abs(f_shifted)))
    # plt.title('fourier space')
    # plt.show()

    # double number of images in fourier space:
    f_filtered = torch.zeros((2*t, h, w, c), dtype=torch.complex64)
    f_filtered[t//2: 3*t//2, :, :, :] = f_shifted
    # plt.imshow(np.log(1 + np.abs(f_filtered)))
    # plt.title('scaled fourier space')
    # plt.show()

    # shift back:
    f_filtered_shifted = torch.fft.fftshift(f_filtered)
    # plt.imshow(np.log(1 + np.abs(f_filtered_shifted)))
    # plt.title('shifted scaled fourier space')
    # plt.show()

    # spatial domain:
    inv_img = torch.fft.ifftn(f_filtered_shifted)  # inverse F.T.
    filtered_img = torch.abs(inv_img)
    filtered_img -= filtered_img.min()
    filtered_img = filtered_img * 255 / filtered_img.max()
    filtered_img = filtered_img.type(torch.uint8)
    # plt.imshow(filtered_img)
    # plt.title('upscaled')
    # plt.show()
    # plt.imsave('upscaled.png', filtered_img)


def double_frames_resizer(x):
    # x = x.transpose((3, 0, 1, 2))
    x = torch.tensor(x, dtype=torch.float32, device='cpu')
    resizer = torch_resizer.Resizer(x.shape, scale_factor=(2, 1, 1, 1),
                                    output_shape=[x.shape[0]*2, x.shape[1], x.shape[2], x.shape[3]],
                                    kernel='cubic', antialiasing=True, device='cpu')
    x_upsampled = resizer(x)
    print('daniel')


if __name__ == '__main__':
    img_path = '/Users/darkushin/Desktop/Project/DeepTemporalSR/Example_data/ground_truth/water/00000.png'
    img = cv2.imread(img_path)[:, :, [2, 1, 0]]  # gray-scale image
    # upscale_im(img)
    img_path2 = '/Users/darkushin/Desktop/Project/DeepTemporalSR/Example_data/ground_truth/water/00010.png'
    img2 = cv2.imread(img_path2)[:, :, [2, 1, 0]]  # gray-scale image
    imgs = np.zeros((2,) + img.shape)
    imgs[0, :, :, :] = img
    imgs[1, :, :, :] = img2
    double_frames_fourier(imgs)
    # double_frames_resizer(imgs)





