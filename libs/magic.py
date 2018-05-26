import io
import cv2
import PIL
import random
import numpy as np
from PIL import ImageEnhance, ImageFilter, Image

applied_arr = []


def pil_to_cv(img):
    open_cv_img = np.array(img)
    open_cv_img = open_cv_img[:, :, ::-1].copy()
    return open_cv_img


def cv_to_pil(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(img)


def convert_to_uint8(image_in):
    temp_image = np.float64(np.copy(image_in))
    cv2.normalize(temp_image, temp_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    return temp_image.astype(np.uint8)


def add_random_motion_blur(img):
    global applied_arr

    cv_img = pil_to_cv(img)
    size = random.randint(1, 6)
    blur_direction = random.randint(0, 3)

    # add to applied string
    applied_arr.append('mblur%i-%i' % (blur_direction, size))

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


def add_random_blur(img):
    global applied_arr
    size = random.randint(0, 2)

    # add to applied string
    applied_arr.append('blur%i' % size)

    return img.filter(ImageFilter.GaussianBlur(radius=random.randint(0, 2)))


def adjust_random_color(img):
    global applied_arr
    adjust = random.randrange(10, 20)
    adjust = adjust / 15

    # add to applied string
    applied_arr.append('color%1.1f' % adjust)

    enhance_image = ImageEnhance.Color(img)
    return enhance_image.enhance(adjust)


def adjust_random_contrast(img):
    global applied_arr
    adjust = random.randrange(10, 20)
    adjust = adjust / 15

    # add to applied string
    applied_arr.append('cont%1.1f' % adjust)

    enhance_image = ImageEnhance.Contrast(img)
    return enhance_image.enhance(adjust)


def adjust_random_brightness(img):
    global applied_arr
    adjust = random.randrange(10, 20)
    adjust = adjust / 15

    # add to applied string
    applied_arr.append('brig%1.1f' % adjust)

    enhance_image = ImageEnhance.Brightness(img)
    return enhance_image.enhance(adjust)


def adjust_random_sharpness(img):
    global applied_arr
    adjust = random.randrange(1, 6)

    # add to applied string
    applied_arr.append('sharp%1.1f' % adjust)

    enhance_image = ImageEnhance.Sharpness(img)
    return enhance_image.enhance(adjust)


def random_jpg_compression(img):
    global applied_arr
    quality = random.randint(20, 90)

    # add to applied string
    applied_arr.append('jpg%i' % quality)

    buffer = io.BytesIO()
    img.save(buffer, format='jpeg', quality=quality)
    return Image.open(buffer)


def flip_horizontally(img):
    global applied_arr
    # add to applied string
    applied_arr.append('flipH')
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def flip_vertically(img):
    global applied_arr
    # add to applied string
    applied_arr.append('flipV')
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def add_random_noise(img):
    global applied_arr

    noise_scale = random.randrange(10, 30)
    noise_scale = noise_scale / 10
    noise_sigma = random.randint(0, 60)
    applied_arr.append('noise%i-%1.1f' % (noise_sigma, noise_scale))

    original_width, original_height = img.size
    width = int(original_width / noise_scale)
    height = int(original_height / noise_scale)
    noise_size = (width, height)
    overlay = Image.new('RGBA', noise_size)
    pix = overlay.load()
    # width, height = img.size
    for x in range(0, width):
        for y in range(0, height):
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                a = random.randint(0, noise_sigma)
                pix[x, y] = (r, g, b, a)

    ret_img = img.copy()
    overlay = overlay.resize(ret_img.size, Image.NEAREST)
    ret_img.paste(overlay, (0, 0), overlay)
    return ret_img


def get_applied_arr():
    global applied_arr
    return applied_arr


def reset_applied_arr():
    global applied_arr
    applied_arr = []


def random_flip(img):
    # flip horizontally 50% of time
    if random.randint(0, 1) == 1:
        img = flip_horizontally(img)

    # flip vertically 10% of time
    if random.randint(1, 10) == 1:
        img = flip_vertically(img)

    return img


def random_magic(img, magic_number=5):
    global applied_arr
    # how often to apply filters, magic number 0-10
    one_in = 11 - magic_number
    # reset applied arr
    applied_arr = []

    # color
    # if random.randint(1, one_in) == 1:
    #     img = adjust_random_color(img)
    #
    # # contrast
    # if random.randint(1, one_in) == 1:
    #     img = adjust_random_contrast(img)
    #
    # # brightness
    # if random.randint(1, one_in) == 1:
    #     img = adjust_random_brightness(img)

    # sharpness
    if random.randint(1, one_in) == 1:
        img = adjust_random_sharpness(img)

    # blur
    if random.randint(1, one_in) == 1:
        img = add_random_blur(img)

    # motion blur
    if random.randint(1, one_in) == 1:
        img = add_random_motion_blur(img)

    # noise
    if random.randint(1, one_in) == 1:
        img = add_random_noise(img)

    # jpg compression, always add
    img = random_jpg_compression(img)

    return img

