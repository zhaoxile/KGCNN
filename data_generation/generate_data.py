import os
import matplotlib.image as img
import numpy as np
import cv2
import random
import h5py
import tool_box.blur as blur
from setting import *
from multiprocessing import Pool

files = os.listdir(clean_path)


def add_rain(img1):
    img2 = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = cv2.split(img2)

    X = np.zeros(Y.shape)
    r_lenth = 15 + 15 * random.random()
    r_lenth = int(r_lenth)
    angle = 45 + 90 * random.random()

    ss = np.random.random()
    if ss < 0.5:
        blur.saltypepper(X, np.random.random() * 0.005 + 0.005)
    elif ss < 0.8:
        blur.saltypepper(X, np.random.random() * 0.005 + 0.095)
    else:
        blur.saltypepper(X, np.random.random() * 0.005 + 0.05)
    kernel = blur.motionblur(r_lenth, angle)
    X = cv2.filter2D(X, -1, blur.gaussianblur([3, 3], np.random.random() * 0.3 + 0.2))
    X = cv2.filter2D(X, -1, kernel)

    Y = X * 255 + Y
    Y[Y > 255] = 255
    img2[:, :, 0] = Y
    img3 = cv2.cvtColor(img2, cv2.COLOR_YCrCb2RGB)
    tmp = np.zeros((SIZE_KERNEL, SIZE_KERNEL))
    row, col = kernel.shape
    tmp[(SIZE_KERNEL - row) // 2: (SIZE_KERNEL - row) // 2 + row,
    (SIZE_KERNEL - col) // 2: (SIZE_KERNEL - col) // 2 + col] = kernel
    return img3, tmp, angle, r_lenth


def generate_date(j):
    Data_rainy = np.zeros((NUM_BATCH, SIZE_INPUT, SIZE_INPUT, 3))
    Data_clean = np.zeros((NUM_BATCH, SIZE_INPUT, SIZE_INPUT, 3))
    Angle = np.zeros((NUM_BATCH, 1))
    Kernel = np.zeros((NUM_BATCH, SIZE_KERNEL, SIZE_KERNEL))
    R_lenth = np.zeros((NUM_BATCH, 1))
    for i in range(NUM_BATCH):
        r_idx = random.randint(0, len(files) - 1)
        img_clean = img.imread(clean_path + files[r_idx])
        if img_clean.dtype != np.uint8:
            img_clean = np.clip(np.uint8(255.0 * img_clean), 0, 255)

        img_rainy, kernel, angle, r_lenth = add_rain(img_clean.copy())

        x = random.randint(0, img_rainy.shape[0] - SIZE_INPUT)
        y = random.randint(0, img_rainy.shape[1] - SIZE_INPUT)

        subim_rainy = img_rainy[x: x + SIZE_INPUT, y: y + SIZE_INPUT, :]
        subim_clean = img_clean[x: x + SIZE_INPUT, y: y + SIZE_INPUT, :]

        Data_rainy[i, :, :, :] = subim_rainy / 255.0
        Data_clean[i, :, :, :] = subim_clean / 255.0
        Kernel[i, :, :] = kernel
        Angle[i, :] = angle
        R_lenth[i, :] = r_lenth

    f = h5py.File('./h5data/train' + str(j + 1) + '.h5', 'w')
    f['rainy'] = Data_rainy
    f['clean'] = Data_clean
    f['kernel'] = Kernel
    f['angle'] = Angle
    f['r_lenth'] = R_lenth
    f.close()
    print(str(j + 1) + '/' + str(NUM_FILES) + ' training h5 files are generated')


if __name__ == '__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool()
    for i in range(NUM_FILES):
        p.apply_async(generate_date, args=(i,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
