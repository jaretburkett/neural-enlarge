from __future__ import print_function, division

from keras.models import Model
from keras.layers import Concatenate, Add, Average, Input, Dense, Flatten, BatchNormalization, Activation, LeakyReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, Convolution2DTranspose
from keras import backend as K
from keras.utils.np_utils import to_categorical
import keras.callbacks as callbacks
import keras.optimizers as optimizers
from libs import img_utils
# import cv2

import numpy as np
import os
import time
import warnings

from libs.advanced import HistoryCheckpoint, SubPixelUpscaling, non_local_block, TensorBoardBatch

try:
    import cv2
    _cv2_available = True
except:
    warnings.warn('Could not load opencv properly. This may affect the quality of output images.')
    _cv2_available = False

def psnr(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Cannot calculate PSNR. Input shapes not same." \
                                         " y_true shape = %s, y_pred shape = %s" % (str(y_true.shape),
                                                                                    str(y_pred.shape))

    return -10. * np.log10(np.mean(np.square(y_pred - y_true)))


def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)


class BaseSuperResolutionModel(object):

    def __init__(self, model_name, scale_factor):
        """
        Base model to provide a standard interface of adding Super Resolution models
        """
        self.model = None  # type: Model
        self.model_name = model_name
        self.scale_factor = scale_factor
        self.weight_path = None

        self.type_scale_type = "norm"  # Default = "norm" = 1. / 255
        self.type_requires_divisible_shape = False
        self.type_true_upscaling = False

        self.evaluation_func = None
        self.uses_learning_phase = False

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128) -> Model:
        """
        Subclass dependent implementation.
        """
        if self.type_requires_divisible_shape and height is not None and width is not None:
            assert height * img_utils._image_scale_multiplier % 4 == 0, "Height of the image must be divisible by 4"
            assert width * img_utils._image_scale_multiplier % 4 == 0, "Width of the image must be divisible by 4"

        if K.image_dim_ordering() == "th":
            if width is not None and height is not None:
                shape = (
                channels, width * img_utils._image_scale_multiplier, height * img_utils._image_scale_multiplier)
            else:
                shape = (channels, None, None)
        else:
            if width is not None and height is not None:
                shape = (
                width * img_utils._image_scale_multiplier, height * img_utils._image_scale_multiplier, channels)
            else:
                shape = (None, None, channels)

        init = Input(shape=shape)

        return init

    def fit(self, batch_size=128, nb_epochs=100, save_history=True, history_fn="Model History.txt") -> Model:
        """
        Standard method to train any of the models.
        """

        samples_per_epoch = img_utils.image_count()
        val_count = img_utils.val_image_count()
        if self.model == None: self.create_model(batch_size=batch_size)

        callback_list = [callbacks.ModelCheckpoint(self.weight_path, monitor='val_PSNRLoss', save_best_only=True,
                                                   mode='max', save_weights_only=True, verbose=2)]
        if save_history:
            callback_list.append(HistoryCheckpoint(history_fn))

            if K.backend() == 'tensorflow':
                log_dir = './%s_logs/' % self.model_name
                tensorboard = TensorBoardBatch(log_dir, batch_size=batch_size)
                callback_list.append(tensorboard)

        print("Training model : %s" % (self.__class__.__name__))
        self.model.fit_generator(img_utils.image_generator(train_path, scale_factor=self.scale_factor,
                                                           small_train_images=self.type_true_upscaling,
                                                           batch_size=batch_size),
                                 steps_per_epoch=samples_per_epoch // batch_size + 1,
                                 epochs=nb_epochs, callbacks=callback_list,
                                 validation_data=img_utils.image_generator(validation_path,
                                                                           scale_factor=self.scale_factor,
                                                                           small_train_images=self.type_true_upscaling,
                                                                           batch_size=batch_size),
                                 validation_steps=val_count // batch_size + 1)

        return self.model

    def evaluate(self, validation_dir):
        if self.type_requires_divisible_shape and not self.type_true_upscaling:
            _evaluate_denoise(self, validation_dir)
        else:
            _evaluate(self, validation_dir)

    def upscale(self, img_path, save_intermediate=False, return_image=False, suffix="scaled",
                patch_size=8, mode="patch", verbose=True):
        """
        Standard method to upscale an image.

        :param img_path:  path to the image
        :param save_intermediate: saves the intermediate upscaled image (bilinear upscale)
        :param return_image: returns a image of shape (height, width, channels).
        :param suffix: suffix of upscaled image
        :param patch_size: size of each patch grid
        :param verbose: whether to print messages
        :param mode: mode of upscaling. Can be "patch" or "fast"
        """
        import os
        from scipy.misc import imread, imresize, imsave

        # Destination path
        path = os.path.splitext(img_path)
        filename = path[0] + "_" + suffix + "(%dx)" % (self.scale_factor) + path[1]

        # Read image
        scale_factor = int(self.scale_factor)
        true_img = imread(img_path, mode='RGB')
        init_dim_1, init_dim_2 = true_img.shape[0], true_img.shape[1]
        if verbose: print("Old Size : ", true_img.shape)
        if verbose: print("New Size : (%d, %d, 3)" % (init_dim_1 * scale_factor, init_dim_2 * scale_factor))

        img_dim_1, img_dim_2 = 0, 0

        if mode == "patch" and self.type_true_upscaling:
            # Overriding mode for True Upscaling models
            mode = 'fast'
            print("Patch mode does not work with True Upscaling models yet. Defaulting to mode='fast'")

        if mode == 'patch':
            # Create patches
            if self.type_requires_divisible_shape:
                if patch_size % 4 != 0:
                    print("Deep Denoise requires patch size which is multiple of 4.\nSetting patch_size = 8.")
                    patch_size = 8

            images = img_utils.make_patches(true_img, scale_factor, patch_size, verbose)

            nb_images = images.shape[0]
            img_dim_1, img_dim_2 = images.shape[1], images.shape[2]
            print("Number of patches = %d, Patch Shape = (%d, %d)" % (nb_images, img_dim_2, img_dim_1))
        else:
            # Use full image for super resolution
            img_dim_1, img_dim_2 = self.__match_autoencoder_size(img_dim_1, img_dim_2, init_dim_1, init_dim_2,
                                                                 scale_factor)

            images = imresize(true_img, (img_dim_1, img_dim_2))
            images = np.expand_dims(images, axis=0)
            print("Image is reshaped to : (%d, %d, %d)" % (images.shape[1], images.shape[2], images.shape[3]))

        # Save intermediate bilinear scaled image is needed for comparison.
        intermediate_img = None
        if save_intermediate:
            if verbose: print("Saving intermediate image.")
            fn = path[0] + "_intermediate_" + path[1]
            intermediate_img = imresize(true_img, (init_dim_1 * scale_factor, init_dim_2 * scale_factor))
            imsave(fn, intermediate_img)

        # Transpose and Process images
        if K.image_dim_ordering() == "th":
            img_conv = images.transpose((0, 3, 1, 2)).astype(np.float32) / 255.
        else:
            img_conv = images.astype(np.float32) / 255.

        model = self.create_model(img_dim_2, img_dim_1, load_weights=True)
        if verbose: print("Model loaded.")

        # Create prediction for image patches
        result = model.predict(img_conv, batch_size=128, verbose=verbose)

        if verbose: print("De-processing images.")

        # Deprocess patches
        if K.image_dim_ordering() == "th":
            result = result.transpose((0, 2, 3, 1)).astype(np.float32) * 255.
        else:
            result = result.astype(np.float32) * 255.

        # Output shape is (original_width * scale, original_height * scale, nb_channels)
        if mode == 'patch':
            out_shape = (init_dim_1 * scale_factor, init_dim_2 * scale_factor, 3)
            result = img_utils.combine_patches(result, out_shape, scale_factor)
        else:
            result = result[0, :, :, :]  # Access the 3 Dimensional image vector

        result = np.clip(result, 0, 255).astype('uint8')

        if _cv2_available:
            # used to remove noisy edges
            result = cv2.pyrUp(result)
            result = cv2.medianBlur(result, 3)
            result = cv2.pyrDown(result)

        if verbose: print("\nCompleted De-processing image.")

        if return_image:
            # Return the image without saving. Useful for testing images.
            return result

        if verbose: print("Saving image.")
        imsave(filename, result)

    def __match_autoencoder_size(self, img_dim_1, img_dim_2, init_dim_1, init_dim_2, scale_factor):
        if self.type_requires_divisible_shape:
            if not self.type_true_upscaling:
                # AE model but not true upsampling
                if ((init_dim_2 * scale_factor) % 4 != 0) or ((init_dim_1 * scale_factor) % 4 != 0) or \
                        (init_dim_2 % 2 != 0) or (init_dim_1 % 2 != 0):

                    print("AE models requires image size which is multiple of 4.")
                    img_dim_2 = ((init_dim_2 * scale_factor) // 4) * 4
                    img_dim_1 = ((init_dim_1 * scale_factor) // 4) * 4

                else:
                    # No change required
                    img_dim_2, img_dim_1 = init_dim_2 * scale_factor, init_dim_1 * scale_factor
            else:
                # AE model and true upsampling
                if ((init_dim_2) % 4 != 0) or ((init_dim_1) % 4 != 0) or \
                        (init_dim_2 % 2 != 0) or (init_dim_1 % 2 != 0):

                    print("AE models requires image size which is multiple of 4.")
                    img_dim_2 = ((init_dim_2) // 4) * 4
                    img_dim_1 = ((init_dim_1) // 4) * 4

                else:
                    # No change required
                    img_dim_2, img_dim_1 = init_dim_2, init_dim_1
        else:
            # Not AE but true upsampling
            if self.type_true_upscaling:
                img_dim_2, img_dim_1 = init_dim_2, init_dim_1
            else:
                # Not AE and not true upsampling
                img_dim_2, img_dim_1 = init_dim_2 * scale_factor, init_dim_1 * scale_factor

        return img_dim_1, img_dim_2,


class DeepDenoiseSR(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(DeepDenoiseSR, self).__init__("Deep Denoise SR", scale_factor)

        # Treat this model as a denoising auto encoder
        # Force the fit, evaluate and upscale methods to take special care about image shape
        self.type_requires_divisible_shape = True

        self.n1 = 64
        self.n2 = 128
        self.n3 = 256

        self.weight_path = "weights/Deep Denoise Weights %dX.h5" % (self.scale_factor)

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        # Perform check that model input shape is divisible by 4
        init = super(DeepDenoiseSR, self).create_model(height, width, channels, load_weights, batch_size)

        c1 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(init)
        c1 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(c1)

        x = MaxPooling2D((2, 2))(c1)

        c2 = Convolution2D(self.n2, (3, 3), activation='relu', padding='same')(x)
        c2 = Convolution2D(self.n2, (3, 3), activation='relu', padding='same')(c2)

        x = MaxPooling2D((2, 2))(c2)

        c3 = Convolution2D(self.n3, (3, 3), activation='relu', padding='same')(x)

        x = UpSampling2D()(c3)

        c2_2 = Convolution2D(self.n2, (3, 3), activation='relu', padding='same')(x)
        c2_2 = Convolution2D(self.n2, (3, 3), activation='relu', padding='same')(c2_2)

        m1 = Add()([c2, c2_2])
        m1 = UpSampling2D()(m1)

        c1_2 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(m1)
        c1_2 = Convolution2D(self.n1, (3, 3), activation='relu', padding='same')(c1_2)

        m2 = Add()([c1, c1_2])

        decoded = Convolution2D(channels, (5, 5), activation='linear', padding='same')(m2)

        model = Model(init, decoded)
        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights(self.weight_path)

        self.model = model
        return model

    def fit(self, batch_size=128, nb_epochs=100, save_history=True, history_fn="Deep DSRCNN History.txt"):
        super(DeepDenoiseSR, self).fit(batch_size, nb_epochs, save_history, history_fn)
