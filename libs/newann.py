from keras import backend as K
import os
from libs.version import __version__
from libs.args import args
from libs.console import error, extend, ansi
import numpy as nd
import collections
from keras.models import Model as KerasModel, load_model as load_keras_model
from keras.layers import Input, Conv2D, Conv2DTranspose, Add, UpSampling2D, MaxPooling2D, Convolution2D, MaxPool2D
from keras import optimizers
from libs.subpixel import SubpixelReshuffleLayer, SubpixDenseUP, SubPixelUpscaling

import numpy as np

from keras.applications.vgg19 import VGG19

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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


class Model(object):

    def __init__(self):
        self.network = collections.OrderedDict()
        self.history = None
        self.model = None
        self.channels = 3

        self.n1 = 50
        self.n2 = 100
        self.n3 = 200

        self.load_model()

    def make_model(self):
        img_size = args.batch_shape / 2
        init = Input((img_size, img_size, 3), name='input_1')
        c1 = Conv2D(self.n1, (3, 3), activation='relu', padding='same')(init)
        c1 = Conv2D(self.n1, (3, 3), activation='relu', padding='same')(c1)

        # 2x output
        c1_up = Conv2DTranspose(self.n3, (3, 3), activation='relu', padding='same', strides=(2, 2))(c1)

        x = c1

        c2 = Conv2D(self.n2, (3, 3), activation='relu', padding='same')(x)
        c2 = Conv2D(self.n2, (3, 3), activation='relu', padding='same')(c2)

        x = MaxPooling2D((2, 2))(c2)

        # c3 = Conv2D(self.n3, (3, 3), activation='relu', padding='same')(x)

        # x = UpSampling2D()(c3)
        c2_2 = Conv2DTranspose(self.n3, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)

        # c2_2 = Conv2D(self.n2, (3, 3), activation='relu', padding='same')(x)
        c2_2 = Conv2D(self.n2, (3, 3), activation='relu', padding='same')(c2_2)

        m1 = Add()([c2, c2_2])
        m1 = Conv2DTranspose(self.n1, (3, 3), activation='relu', padding='same', strides=(2, 2))(m1)
        # m1 = UpSampling2D()(m1)

        c1_2 = Conv2D(self.n2, (3, 3), activation='relu', padding='same')(m1)
        c1_2 = Conv2D(self.n3, (3, 3), activation='relu', padding='same')(c1_2)

        m2 = Add()([c1_up, c1_2])

        decoded = Conv2D(self.channels, (5, 5), activation='linear', padding='same')(m2)

        model = KerasModel(init, decoded)
        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])

        model.summary()

        # load weights
        if os.path.exists(self.get_filename(absolute=True)):
            print('Importing weights from file %s' % self.get_filename(absolute=False))
            model.load_weights(self.get_filename(absolute=True), by_name=True)
        self.model = model

        # return model

    def fit(self, images, seeds):
        # print('fitting')
        # self.history = self.model.fit(seeds, images, verbose=0, epochs=1, validation_split=.2)
        # print('seeds', seeds.shape)
        # print('images', images.shape)
        self.history = self.model.train_on_batch(seeds, images)
        # return self.history.history
        # print(self.history)
        return self.history

    def get_filename(self, absolute=False):
        filename = 'models/ne%ix-%s-%s-%s.h5' % (args.zoom, args.type, args.model, __version__)
        return os.path.abspath(filename) if absolute else filename

    def save(self):
        print('Saving model to %s' % self.get_filename())
        self.model.save(self.get_filename(absolute=True))

    def load_model(self):
        if not os.path.exists(self.get_filename(absolute=True)):
            if args.train:
                self.make_model()
            else:
                self.make_model()
                # error("Model file with pre-trained convolution layers not found. Download it here...",
                #       "https://github.com/jaretburkett/neural-enlarge/releases/download/v%s/%s" % (
                #           __version__, self.get_filename()))
        else:
            if args.train:
                self.make_model()
            else:
                print('Loading model %s' % self.get_filename())
                self.model = load_keras_model(
                    self.get_filename(absolute=True),
                    custom_objects={
                        'PSNRLoss': PSNRLoss,
                        'SubPixelUpscaling': SubPixelUpscaling
                    })
                print('Model Loaded')


    def set_learning_rate(self, lr):
        K.set_value(self.model.optimizer.lr, lr)

    def predict(self, img_arr):
        scald = []
        repro = []
        for x in img_arr:
            scald.append(x)
            # training_seeds = np.transpose(x, (2, 1, 0))  # This is correct, not sure why it is flipped
            training_seeds = np.transpose(x, (1, 2, 0))
            processed = self.model.predict(nd.array([training_seeds]))
            repro.append(np.transpose(processed[0], (2, 0, 1)))

        return scald, repro

    def output_per_layer(self, img_arr):

        each_layer = []
        layer_names = [layer.name for layer in self.model.layers]

        inp = self.model.input  # input placeholder
        outputs = [layer.output for layer in self.model.layers]  # all layer outputs
        functor = K.function([inp] + [K.learning_phase()], outputs)  # evaluation function

        # test_img = np.transpose(img_arr[0], (2, 1, 0))  # This is correct, not sure why it is flipped
        test_img = np.transpose(img_arr[0], (1, 2, 0))
        test = nd.array([test_img])
        layer_outs = functor([test, 1.])
        each_layer = layer_outs
        return layer_names, each_layer


class VGG19b34(Model):

    def make_model(self):
        img_size = args.batch_shape / 2
        init = Input((img_size, img_size, 3), name='input_1')
        # init = Input((100, 100, 3), name='input_1')

        vgg19 = VGG19(include_top=False, input_tensor=init) # from 100 to 3
        vgg19_layer_dict = dict([(layer.name, layer) for layer in vgg19.layers])
        block1_conv2 = vgg19_layer_dict['block1_conv2'].output  # 100 - 64
        block2_conv2 = vgg19_layer_dict['block2_conv2'].output  # 50 - 128
        block3_conv4 = vgg19_layer_dict['block3_conv4'].output  # 25 - 256

        # 25
        b4_1 = Conv2DTranspose(64, (2, 2), activation='relu', padding='same', name='block4_up_1', strides=(4, 4))(block3_conv4)
        # 100

        # 100
        b4 = Add(name='block4_join')([b4_1, block1_conv2])

        # b5_1 = Conv2DTranspose(64, (2, 2), activation='relu', padding='same', name='block5_up_1', strides=(2, 2))(b4)
        b5_2 = Conv2DTranspose(64, (2, 2), activation='relu', padding='same', name='block5_up_1', strides=(2, 2))(block2_conv2)
        # 100

        # b5 = Add(name='block5_join')([b4, b5_2])
        b6 = Add(name='block6_join')([b5_2, b4])
        b6_up = Conv2DTranspose(64, (2, 2), activation='relu', padding='same', name='block6_up_1', strides=(2, 2))(b6)

        decoded = Conv2D(self.channels, (5, 5), activation='linear', padding='same', name='out_1')(b6_up)

        model = KerasModel(init, decoded)
        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])

        model.summary()

        # load weights
        if os.path.exists(self.get_filename(absolute=True)):
            print('Importing weights from file %s' % self.get_filename(absolute=False))
            model.load_weights(self.get_filename(absolute=True), by_name=True)
        self.model = model

        # return model


class VGG19b13(Model):

    def make_model(self):
        img_size = args.batch_shape / 2
        init = Input((img_size, img_size, self.channels), name='input_1')
        # init = Input((100, 100, 3), name='input_1')

        # Block 1 VGG19 -100
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(init)
        b1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(b1)

        # Block 2 -50
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)

        # 50
        b3_1 = Conv2DTranspose(256, (2, 2), activation='relu', padding='same', name='block2_up', strides=(4, 4))(x)
        # 200

        # 100
        b3_2 = Conv2DTranspose(256, (2, 2), activation='relu', padding='same', name='block1_up', strides=(2, 2))(b1)
        # 200

        b3 = Add(name='block3_join')([b3_1, b3_2])

        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(b3)

        decoded = Conv2D(self.channels, (5, 5), activation='linear', padding='same', name='out_1')(x)

        model = KerasModel(init, decoded)
        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        model.summary()

        # preload existing weights, these will get overwritten when we load existing training weights
        vgg19 = VGG19(include_top=False, input_shape=(img_size, img_size, 3))  # from 100 to 3
        vgg19_layer_dict = dict([(layer.name, layer) for layer in vgg19.layers])
        model_layer_dict = dict([(layer.name, layer) for layer in model.layers])
        model_layer_dict['block1_conv1'].set_weights(vgg19_layer_dict['block1_conv1'].get_weights())
        model_layer_dict['block1_conv2'].set_weights(vgg19_layer_dict['block1_conv2'].get_weights())
        model_layer_dict['block1_pool'].set_weights(vgg19_layer_dict['block1_pool'].get_weights())
        model_layer_dict['block2_conv1'].set_weights(vgg19_layer_dict['block2_conv1'].get_weights())
        model_layer_dict['block2_conv2'].set_weights(vgg19_layer_dict['block2_conv2'].get_weights())


        # load weights
        if os.path.exists(self.get_filename(absolute=True)):
            print('Importing weights from file %s' % self.get_filename(absolute=False))
            model.load_weights(self.get_filename(absolute=True), by_name=True)
        self.model = model

        # return model


class DEBLUR(Model):

    def make_model(self):
        img_size = args.batch_shape / args.zoom
        init = Input((img_size, img_size, self.channels), name='input_1')
        # init = Input((100, 100, 3), name='input_1')

        # Block 1 VGG19 -100
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(init)
        b1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(b1)

        # Block 2 -50
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)

        # 50
        b3_1 = Conv2DTranspose(64, (2, 2), activation='relu', padding='same', name='block2_up', strides=(2, 2))(x)
        # 100

        b3 = Add(name='block3_join')([b3_1, b1])

        x = Conv2D(128, (5, 5), activation='relu', padding='same', name='block3_conv1')(b3)
        x = Conv2D(256, (5, 5), activation='relu', padding='same', name='block3_conv2')(x)
        decoded = Conv2D(self.channels, (5, 5), activation='linear', padding='same', name='out_1')(x)

        model = KerasModel(init, decoded)
        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        model.summary()

        # preload existing weights, these will get overwritten when we load existing training weights
        vgg19 = VGG19(include_top=False, input_shape=(img_size, img_size, 3))  # from 100 to 3
        vgg19_layer_dict = dict([(layer.name, layer) for layer in vgg19.layers])
        model_layer_dict = dict([(layer.name, layer) for layer in model.layers])
        model_layer_dict['block1_conv1'].set_weights(vgg19_layer_dict['block1_conv1'].get_weights())
        model_layer_dict['block1_conv2'].set_weights(vgg19_layer_dict['block1_conv2'].get_weights())
        model_layer_dict['block1_pool'].set_weights(vgg19_layer_dict['block1_pool'].get_weights())
        model_layer_dict['block2_conv1'].set_weights(vgg19_layer_dict['block2_conv1'].get_weights())
        model_layer_dict['block2_conv2'].set_weights(vgg19_layer_dict['block2_conv2'].get_weights())


        # load weights
        if os.path.exists(self.get_filename(absolute=True)):
            print('Importing weights from file %s' % self.get_filename(absolute=False))
            model.load_weights(self.get_filename(absolute=True), by_name=True)
        self.model = model

        # return model
