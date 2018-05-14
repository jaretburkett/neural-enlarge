import cv2
import random
import numpy as np
import PIL
from PIL import ImageEnhance


def pil_to_cv(img):
    open_cv_img = np.array(img)
    open_cv_img = open_cv_img[:, :, ::-1].copy()
    return open_cv_img


def cv_to_pil(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(img)


def add_random_motion_blur(img):

    cv_img = pil_to_cv(img)
    size = random.randint(1, 10)
    blur_direction = random.randint(0, 3)

    kernel_motion_blur = np.zeros((size, size))

    # horizontal
    if blur_direction == 0:
        kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)

    # vertical
    if blur_direction == 1:
        i = 0
        while i < len(kernel_motion_blur):
            kernel_motion_blur[i][int((size - 1) / 2)] = np.ones(1)
            i += 1

    # top left to bottom right
    if blur_direction == 2:
        i = 0
        while i < len(kernel_motion_blur):
            kernel_motion_blur[i][i] = np.ones(1)
            i += 1

    # top right to bottom left
    if blur_direction == 3:
        i = 0
        while i < len(kernel_motion_blur):
            kernel_motion_blur[i][size - i - 1] = np.ones(1)
            i += 1

    kernel_motion_blur = kernel_motion_blur / size

    # applying the kernel to the input image
    cv_img = cv2.filter2D(cv_img, -1, kernel_motion_blur)
    return cv_to_pil(cv_img)


def adjust_random_color(img):
    adjust = random.randrange(10, 20)
    adjust = adjust / 15
    contrast_image = ImageEnhance.Color(img)
    return contrast_image.enhance(adjust)


def adjust_random_contrast(img):
    adjust = random.randrange(10, 20)
    adjust = adjust / 15
    contrast_image = ImageEnhance.Contrast(img)
    return contrast_image.enhance(adjust)


def adjust_random_brightness(img):
    adjust = random.randrange(10, 20)
    adjust = adjust / 15
    contrast_image = ImageEnhance.Brightness(img)
    return contrast_image.enhance(adjust)


def adjust_random_sharpness(img):
    adjust = random.randrange(0, 15)
    contrast_image = ImageEnhance.Sharpness(img)
    return contrast_image.enhance(adjust)


def random_magic(img):
    one_in = 3

    # color
    if random.randint(1, one_in) == 1:
        img = adjust_random_color(img)

    # contrast
    if random.randint(1, one_in) == 1:
        img = adjust_random_contrast(img)

    # brightness
    if random.randint(1, one_in) == 1:
        img = adjust_random_brightness(img)

    # sharpness
    if random.randint(1, one_in) == 1:
        img = adjust_random_sharpness(img)

    # motion blur
    if random.randint(1, one_in) == 1:
        img = add_random_motion_blur(img)

    return img

