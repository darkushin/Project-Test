import utils
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import torch_resizer
import torch
import augmentations
import glob


def display_frames():
    data_path = '/Users/darkushin/Desktop/Project/DeepTemporalSR/results/output/eagle-no_training'
    prefix = ''
    dtype = 'float32'

    video_tensor = np.asarray(utils.read_seq_from_folder(data_path, prefix, dtype))
    # print(video_tensor.shape)
    plt.imshow(video_tensor[300, :, :, :])
    plt.show()


def swap_axes(data_path, output_dir):
    prefix = ''
    dtype = 'float32'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if not os.path.isdir(data_path):
        raise Exception('Invalid data path')

    video_tensor = np.asarray(utils.read_seq_from_folder(data_path, prefix, dtype))
    for x in range(video_tensor.shape[2]):
        plt.imsave(f'{output_dir}/{x:05d}.png', video_tensor[:, :, x, :].transpose(1, 0, 2))


def save_frames():
    vidcap = cv2.VideoCapture('/Users/darkushin/Desktop/Project/DeepTempo/horse-small.mp4')
    success, image = vidcap.read()
    count = 0
    while success:
        if not count % 100: print(count)
        cv2.imwrite(f"/Users/darkushin/Desktop/Project/DeepTemporalSR/Example_data/ground_truth/Horse"
                    f"/{count:05d}.png", image)  # save frame as JPEG file
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1


def create_SLR_video():
    orig_data_path = 'Example_data/ground_truth/Horse-short'
    orig_tensor = np.asarray(utils.read_seq_from_folder(orig_data_path, '', 'float32'))
    assert orig_tensor.shape[1] % 2 == orig_tensor.shape[
        2] % 2 == 0, f'assertion error in downscale_for_BP: video shape not divisible by needed downscale'
    working_dir = ''

    folder_name = os.path.join(working_dir, 'Example_data/blur8/Horse_short_lr_x')
    os.mkdir(folder_name)

    resizer = torch_resizer.Resizer(orig_tensor.shape,
                                    output_shape=[int(orig_tensor.shape[0]),
                                                  int(orig_tensor.shape[1]),
                                                  int(orig_tensor.shape[2] / 2),
                                                  orig_tensor.shape[3]],
                                    kernel='cubic', antialiasing=True, device='cpu',
                                    dtype=torch.float32)  # todo: was `dtype=torch.float16`
    resized_tensor = np.clip(resizer(torch.tensor(orig_tensor, dtype=torch.float16).to('cpu')).cpu().numpy(), 0., 1.)

    utils.save_output_result(resized_tensor, folder_name)


def hr_to_lr(hr_tensor, jump=2):
    """
    take a HR tensor and return its (temporally) LR tensor, in the manner determined in config.
    :param hr_tensor: np array
    :return: none, plots the frames or tensors
    """
    # check that the HR tensor is [F,H,W,C]
    assert len(
        hr_tensor.shape) == 4, f'assert error in hr_to_lr.HR tensor shape len is {len(hr_tensor.shape)},not 4'

    lr_tensor = augmentations.blur_sample_tensor(hr_tensor, 0, sample_jump=jump, blur_flag=True)
    return lr_tensor


def subsample_time(data_path, output_folder):
    prefix = ''
    dtype = 'float32'
    os.mkdir(output_folder)

    video_tensor = np.asarray(utils.read_seq_from_folder(data_path, prefix, dtype))
    lr_tensor = hr_to_lr(hr_tensor=video_tensor)
    print('daniel')


def create_video(dataset_name, frames_folder='results/output', output_videos_path='results/videos'):
    """
    Create a video from the given frames.
    """
    frames_path = sorted(glob.glob(os.path.join(frames_folder, dataset_name, '*')))
    if len(frames_path) == 0:
        raise Exception(f'No such folder: {os.path.join(frames_folder, dataset_name)}')
    im_shape = cv2.imread(frames_path[0]).shape
    if not os.path.isdir(output_videos_path):
        os.mkdir(output_videos_path)
    out = cv2.VideoWriter(os.path.join(output_videos_path, f'{dataset_name}.avi'),
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, (im_shape[1], im_shape[0]))
    print("Starting to save images to output video")
    for frame in frames_path:
        im = cv2.imread(frame)
        out.write(im)

    print("Done creating video")
    out.release()


def create_multiple_videos(dataset):
    frame_folders = get_test_folders(dataset)
    for frames in frame_folders:
        create_video(frames, f'/Users/darkushin/Desktop/Project/transfer-files/transfer-files/Original_Tests/{dataset}',
                     f'results/videos/{dataset}')


def compute_psnr(imgs1, imgs2):
    """
    Compute the average PSNR value for the given two sets of images
    """
    num_imgs = len(imgs1)
    average_psnr = 0
    for i in range(num_imgs):
        im1 = cv2.imread(imgs1[i])
        im2 = cv2.imread(imgs2[i])
        average_psnr += cv2.PSNR(im1, im2) / num_imgs
    return average_psnr


def get_test_folders(dataset):
    return [f'output-{dataset}_5_epochs_train', f'output-{dataset}_full_train',
            f'output-{dataset}_within_only', f'output-{dataset}_across_only',
            f'output-{dataset}_10k_epochs_within_only', f'output-{dataset}_10k_both',
            f'output-{dataset}_10k_across_only',
            f'output-{dataset}_1000_epochs_within_only', f'output-{dataset}_resizer_only']


def psnr(dataset):
    test_folders = get_test_folders(dataset)
    # compare every two frames and output the mean of all frames:
    GT_imgs = sorted(glob.glob(os.path.join('Example_data/ground_truth', dataset, '*')))
    results = []
    for test in test_folders:
        restored_imgs = sorted(glob.glob(os.path.join(
            f'/Users/darkushin/Desktop/Project/transfer-files/transfer-files/Original_Tests/{dataset}',
                                                      test, '*')))
        if len(restored_imgs) == 0:
            print(f'Not running on file: {test}!!!')
            results.append(0)
            continue
        average_psnr = compute_psnr(GT_imgs, restored_imgs)
        results.append(average_psnr)

    output = f'Dataset: {dataset}\n' \
             f'5 Epochs Train: {results[0]}\n' \
             f'Full Train: {results[1]}\n'\
             f'Within Only: {results[2]}\n'\
             f'Across Only: {results[3]}\n'\
             f'10k Epochs Within Only: {results[4]}\n'\
             f'10k Epochs Both: {results[5]}\n'\
             f'10k Epochs Across Only: {results[6]}\n'\
             f'1000 Epochs Within Only: {results[7]}\n'\
             f'Resizer Only: {results[8]}\n'


    print(output)


# save_frames()
# display_frames()
# create_SLR_video()
# swap_axes('/Users/darkushin/Desktop/Project/DeepTemporalSR/Example_data/tests/bus_hr_x_with_tps',
#           'Example_data/tests/bus_returned_hr_x_with_tps')
# subsample_time('Example_data/ground_truth/Horse', 'Example_data/ground_truth/Horse-short')
# create_video('christmas', '/Users/darkushin/Desktop/Project/DeepTemporalSR/Example_data/blur8/')
psnr('christmas')
# create_multiple_videos('short_billiard')

# GT_imgs = sorted(glob.glob(os.path.join('Example_data/ground_truth', 'christmas', '*')))
# full_train = sorted(glob.glob(os.path.join(
#     '/Users/darkushin/Desktop/Project/transfer-files/transfer-files/Original_Tests/christmas/output'
#     '-christmas_full_train', '*')))
# full_train2 = sorted(glob.glob(os.path.join(
#     '/Users/darkushin/Desktop/Project/transfer-files/transfer-files/Original_Tests/christmas/output'
#     '-christmas_full_train_2', '*')))
# print(compute_psnr(full_train, full_train2))



