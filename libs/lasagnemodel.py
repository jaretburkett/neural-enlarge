
import os
import numpy as np
from libs.version import __version__
from libs.console import error, extend, ansi
import keras
import collections
import tensorflow as tf
import pickle
import bz2
from libs.console import extend
from libs.newann import Model
from libs.subpixel import SubPixelUpscaling
import keras.backend as K
from keras import optimizers
from keras import Model as KerasModel
from keras.layers import Input, Concatenate, Conv2D, ZeroPadding2D, PReLU, Add, Layer, Conv2DTranspose, Reshape
from libs.args import args


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


class LasagneModel(Model):

    def make_model(self):

        self.network = collections.OrderedDict()
        wh = args.batch_shape / 2
        self.network['img'] = Input((wh, wh, 3), name='img')
        if args.train:
            wh = args.batch_shape / 2
            self.network['seed'] = Input((wh, wh, 3), name='seed')
        else:
            wh = args.rendering_tile + (args.rendering_overlap * 2)
            self.network['seed'] = Input((wh, wh, 3), name='seed')

        config, params = self.load_lasagne_model()

        input = self.last_layer()

        for key, value in config.items():
            # sets config as args ex args[key] = value
            setattr(args, key, value)

        # set arg zoom for the loaded model
        args.zoom = 2 ** (args.generator_upscale - args.generator_downscale)

        units_iter = extend(args.generator_filters)
        units = next(units_iter)

        self.make_layer('iter_0', input, units, filter_size=(7, 7), pad=(3, 3))

        for i in range(0, args.generator_downscale):
            self.make_layer('downscale%i' % i, self.last_layer(), next(units_iter), filter_size=(4, 4), stride=(2, 2))

        units = next(units_iter)
        for i in range(0, args.generator_blocks):
            self.make_block('iter_%i' % (i + 1), self.last_layer(), units)

        # need to reshape
        #     self.network['reshape_1'] = Reshape((128, 128, units // 3 * 3))(self.last_layer())
        for i in range(0, args.generator_upscale):
            u = next(units_iter)
            self.make_layer('upscale%i_2' % i, self.last_layer(), u * 4)
            # self.network['upscale%i_1' % i] = SubpixelReshuffleLayer(u, 2, name='upscale%i_1' % i)(self.last_layer())
            self.network['upscale%i_1' % i] = SubPixelUpscaling(2, u, name='upscale%i_1' % i)(self.last_layer())

        padding = ZeroPadding2D(padding=(3, 3), name='out_')(self.last_layer())
        self.network['out'] = Conv2D(3, kernel_size=(7, 7), name='out')(padding)

        # todo skip for now
        # if args.train:
        #     concatenated = Concatenate([self.network['img'], self.network['out']], axis=0)
        #     self.setup_perceptual(concatenated)
        #     self.load_perceptual()
        #     self.setup_discriminator()

        input_layers = input
        output_layers = self.network['out']

        self.model = KerasModel(input_layers, output_layers)
        # todo change optimizer
        adam = optimizers.Adam(lr=1e-3)
        self.model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        self.model.summary()

        # load weights

        self.load_weights(params)
        self.save()
        # self.compile()


    def load_weights(self, params):
        layer_dict = dict([(layer.name, layer) for layer in self.model.layers])
        for key, value in params.items():
            fixed_val = value
            if len(params[key]) == 2:
                # fix ordering of weights
                fixed_val[0] = np.transpose(params[key][0], (3, 2, 1, 0))

            if '>' in key:
                # adjust prelu shape
                fixed_val[0] = params[key][0][np.newaxis,np.newaxis,:]
            # keras does not like layer names with some symbols
            new_key = key.replace('>', '_pr')
            new_key = new_key.replace('.', '_')
            print('Loading weights for %s' % new_key)
            layer_dict[new_key].set_weights(fixed_val)

    def last_layer(self):
        return list(self.network.values())[-1]

    def make_layer(self, name, input, units, filter_size=(3, 3), stride=(1, 1), pad=(1, 1), alpha=0.25):
        padding = ZeroPadding2D(padding=pad, name=name + '_')(input)
        # todo should this be same padding?
        conv = Conv2D(filters=units, kernel_size=filter_size, strides=stride, padding='valid', name=name + 'x')(padding)
        prelu = PReLU(alpha_initializer=keras.initializers.Constant(alpha), shared_axes=[1, 2], name=name + '_pr')(conv)
        # self.network[name + '_'] = padding
        self.network[name + 'x'] = conv
        self.network[name + '_pr'] = prelu
        return prelu

    def make_block(self, name, input, units):
        self.make_layer(name + '-A', input, units, alpha=0.1)
        return Add()([input, self.last_layer()]) if args.generator_residual else self.last_layer()

    def load_lasagne_model(self):
        if not os.path.exists(self.get_filename_lasagne(absolute=True)):
            if args.train:
                return {}, {}
            error("Model file with pre-trained convolution layers not found. Download it here...",
                  "https://github.com/jaretburkett/neural-enlarge/releases/download/v%s/%s" % (
                  __version__, self.get_filename_lasagne()))
        print('  - Loaded file `{}` with trained model.'.format(self.get_filename_lasagne()))
        return pickle.load(bz2.open(self.get_filename_lasagne(absolute=True), 'rb'))

    def get_filename_lasagne(self, absolute=False):
        filename = 'models/ne%ix-%s-%s-%s.pkl.bz2' % (args.zoom, args.type, args.model, __version__)
        return os.path.abspath(filename) if absolute else filename


#%%
if __name__ == "__main__":
    with tf.device('/CPU:0'):
        model = LasagneModel()

