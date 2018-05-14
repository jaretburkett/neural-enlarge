import os
import cv2
import random
import numpy as np
import PIL


def pil_to_cv(img):
    open_cv_img = np.array(img)
    open_cv_img = open_cv_img[:, :, ::-1].copy()
    return open_cv_img


def cv_to_pil(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(img)


def add_random_motion_blur(img):

    cv_img = pil_to_cv(img)
    size = random.randint(1, 30)
    type = random.randint(0, 3)

    kernel_motion_blur = np.zeros((size, size))

    # horizontal
    if type == 0:
        kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)

    # verticle
    if type == 1:
        i = 0
        while i < len(kernel_motion_blur):
            kernel_motion_blur[i][int((size - 1) / 2)] = np.ones(1)
            i += 1

    # top left to bottom right
    if type == 2:
        i = 0
        while i < len(kernel_motion_blur):
            kernel_motion_blur[i][i] = np.ones(1)
            i += 1

    # top right to bottom left
    if type == 3:
        i = 0
        while i < len(kernel_motion_blur):
            kernel_motion_blur[i][size - i - 1] = np.ones(1)
            i += 1

    kernel_motion_blur = kernel_motion_blur / size

    # applying the kernel to the input image
    cv_img = cv2.filter2D(cv_img, -1, kernel_motion_blur)
    return cv_to_pil(cv_img)


def random_magic(img):
    return add_random_motion_blur(img)

