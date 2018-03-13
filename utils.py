import logging

from collections import Counter
from os import listdir
from os.path import isdir, join


import numpy as np

from PIL import Image
from visdom import Visdom

IMG_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp',
]

RESOLUTIONS = {
    0: [[1520,2688]], # flips
    1: [[3264,2448]], # no flips
    2: [[2432,4320]], # flips
    3: [[3120,4160]], # flips
    4: [[4128,2322]], # no flips
    5: [[3264,2448]], # no flips
    6: [[3024,4032]], # flips
    7: [[1040,780],  # Motorola-Nexus-6 no flips
        [3088,4130], [3120,4160]], # Motorola-Nexus-6 flips
    8: [[4128,2322]], # no flips
    9: [[6000,4000]], # no flips
}

ORIENTATION_FLIP_ALLOWED = [
    True,
    False,
    True,
    True,
    False,
    False,
    True,
    True,
    False,
    False
]

LABELS = ['HTC-1-M7',
 'iPhone-4s',
 'iPhone-6',
 'LG-Nexus-5x',
 'Motorola-Droid-Maxx',
 'Motorola-Nexus-6',
 'Motorola-X',
 'Samsung-Galaxy-Note3',
 'Samsung-Galaxy-S4',
 'Sony-NEX-7']

LABEL_DICT = {name: i for (i, name) in enumerate(LABELS)}

for key, resolutions in RESOLUTIONS.copy().items():
    resolutions.extend([resolution[::-1] for resolution in resolutions])
    RESOLUTIONS[key] = resolutions

MANIPULATIONS = ['jpg70', 'jpg80', 'jpg90', 'gamma0.8', 'gamma0.9', 'gamma1.1', 'gamma1.2', 'bicubic0.5', 'bicubic0.8', 'bicubic1.5', 'bicubic2.0']

def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in IMG_EXTENSIONS)


import cv2


def random_crop_fft(img, W):
    nr, nc = img.shape[:2]
    r1, c1 = np.random.randint(nr - W), np.random.randint(nc - W)
    imgc = img[r1:r1 + W, c1:c1 + W, :]

    img1 = imgc - cv2.GaussianBlur(imgc, (3, 3), 0)
    imgs1 = np.sum(img1, axis=2)

    sf = np.stack([
        np.fft.fftshift(np.fft.fft2(imgs1)),
        np.fft.fftshift(np.fft.fft2(img1[:, :, 0] - img1[:, :, 1])),
        np.fft.fftshift(np.fft.fft2(img1[:, :, 1] - img1[:, :, 2])),
        np.fft.fftshift(np.fft.fft2(img1[:, :, 2] - img1[:, :, 0]))], axis=-1)
    return np.abs(sf)


def imread_residual_fft(fn, W, navg):
    # print(fn, rss())
    img = cv2.imread(fn).astype(np.float32) / 255.0
    return sum(map(lambda x: random_crop_fft(img, W), range(navg))) / navg


def noise_pattern(modelname, W, navg=256):
    files = train.path[train.modelname == modelname].values
    orientations = np.vectorize(is_landscape)(files)
    if np.sum(orientations) < len(orientations) // 2:
        orientations = ~orientations
    files = files[orientations]

    from multiprocess import Pool
    with Pool() as pool:
        s = sum(tqdm.tqdm(pool.imap(lambda fn: imread_residual_fft(fn, W, navg), files), total=len(files),
                          desc=modelname)) / len(files)

    return s

def plot_model_features(modelname, W):
    s = noise_pattern(modelname, W)
    nchans = s.shape[2]
    nrows = (nchans + 3) // 4
    _, ax = plt.subplots(nrows, 4, figsize=(16, 4 * nrows))
    ax = ax.flatten()

    for c in range(nchans):
        eps = np.max(s[:, :, c]) * 1e-2
        s1 = np.log(s[:, :, c] + eps) - np.log(eps)
        img = (s1 * 255 / np.max(s1)).astype(np.uint8)
        ax[c].imshow(cv2.equalizeHist(img))

    for ax1 in ax[nchans:]:
        ax1.axis('off')

    plt.show()


def plot_all_model_features(W):
    print("Feature Size={}".format(W))
    for modelname in train.modelname.unique():
        plot_model_features(modelname, W)