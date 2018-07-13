import os
import bz2
import keras
import pickle
import collections
import numpy as np
import tensorflow as tf
from keras import Model
from libs.args import args
from keras import optimizers
from keras.optimizers import Adam
from keras import losses
from libs.console import error
from libs.console import extend
from libs.losses import PSNRLoss
from libs.version import __version__
from libs.layers import SubPixelUpscaling
from libs.model_classes.base import BaseModel
from keras.layers import Input, Concatenate, Conv2D, ZeroPadding2D, PReLU, Add
from keras.layers import Lambda, MaxPooling2D, BatchNormalization, UpSampling2D

from keras import layers
import keras.backend as K
from keras.applications.vgg19 import VGG19


class LasagneModel(BaseModel):
    def __init__(self):
        super(LasagneModel, self).__init__()
        self.model_generator = None
        self.model_discriminator = None
        self.model_adversarial = None
        self.model_perceptual = None
        self.network = collections.OrderedDict()
        self.gen_lr = 0.0
        self.adversary_weight = 0.0

    def make_model(self):
        # config, params = self.load_lasagne_model()
        config = self.load_config()
        for key, value in config.items():
            # sets config as args ex args[key] = value
            setattr(args, key, value)

        # set arg zoom for the loaded model
        args.zoom = 2 ** (args.generator_upscale - args.generator_downscale)

        wh = args.batch_shape
        wh_h = wh / args.zoom
        self.network['img'] = Input((wh, wh, 3), name='img')
        if args.train:
            self.network['seed'] = Input((wh_h, wh_h, 3), name='seed')
        else:
            wh = args.rendering_tile + (args.rendering_overlap * 2)
            self.network['seed'] = Input((wh, wh, 3), name='seed')

        self.setup_generator(self.last_layer(), config)

        if args.train:
            self.network['percept_input'] = Input((wh, wh, 3), name='percept_input')
            self.setup_perceptual()
            self.load_perceptual()
            self.setup_discriminator()
        self.compile()

    def load_weights(self, params):
        layer_dict = dict([(layer.name, layer) for layer in self.model.layers])
        for key, value in params.items():
            fixed_val = value
            if len(params[key]) == 2:
                # fix ordering of weights (from lasagne to keras)
                fixed_val[0] = np.transpose(params[key][0], (3, 2, 1, 0))
            if '>' in key:
                # adjust prelu shape
                fixed_val[0] = params[key][0][np.newaxis, np.newaxis, :]
            # keras does not like layer names with some symbols
            new_key = key.replace('>', '_pr')
            new_key = new_key.replace('.', '_')
            print('Loading weights for %s' % new_key)
            layer_dict[new_key].set_weights(fixed_val)

    def last_layer(self):
        return list(self.network.values())[-1]

    def make_layer(self, name, input, units, filter_size=(3, 3), stride=(1, 1), pad=(1, 1), alpha=0.25):
        padding = ZeroPadding2D(padding=pad, name=name + '_')(input)
        conv = Conv2D(filters=units, kernel_size=filter_size, strides=stride, padding='valid', name=name + 'x')(padding)
        prelu = PReLU(alpha_initializer=keras.initializers.Constant(alpha), shared_axes=[1, 2], name=name + '_pr')(conv)
        self.network[name + 'x'] = conv
        self.network[name + '_pr'] = prelu
        return prelu

    def make_block(self, name, input, units):
        self.make_layer(name + '-A', input, units, alpha=0.1)
        return Add()([input, self.last_layer()]) if args.generator_residual else self.last_layer()

    def setup_generator(self, input, config):

        units_iter = extend(args.generator_filters)
        units = next(units_iter)

        self.make_layer('iter_0', input, units, filter_size=(7, 7), pad=(3, 3))

        for i in range(0, args.generator_downscale):
            self.make_layer('downscale%i' % i, self.last_layer(), next(units_iter), filter_size=(4, 4), stride=(2, 2))

        units = next(units_iter)
        for i in range(0, args.generator_blocks):
            self.make_block('iter_%i' % (i + 1), self.last_layer(), units)

        # need to reshape
        for i in range(0, args.generator_upscale):
            u = next(units_iter)
            self.make_layer('upscale%i_2' % i, self.last_layer(), u * 4)
            # self.network['upscale%i_1' % i] = SubPixelUpscaling(2, u, name='upscale%i_1' % i)(self.last_layer())
            self.network['upscale%i_1' % i] = UpSampling2D(name='upscale%i_1' % i)(self.last_layer())

        padding = ZeroPadding2D(padding=(3, 3), name='out_')(self.last_layer())
        self.network['out'] = Conv2D(3, kernel_size=(7, 7), name='out')(padding)

    def setup_perceptual(self):
        # build vgg19 perceptual layers
        def ConvLayer(input_layer, units, kernel, name):
            return Conv2D(units, (kernel, kernel),
                          activation='relu',
                          padding='same',
                          name=name)(self.network['percept'])(input_layer)

        def PoolLayer(input_layer, pool_size, name):
            return MaxPooling2D((pool_size, pool_size), strides=(2, 2), name=name)(input_layer)

        self.network['percept'] = Lambda(lambda x: ((x + 0.5) * 255.0) - K.variable(
            np.array([103.939, 116.779, 123.680], dtype=np.float32).reshape((1, 1, 1, 3))))(self.network['percept_input'])

        self.network['mse'] = self.network['percept']
        self.network['conv1_1'] = ConvLayer(self.network['percept'], 64, 3, name='block1_conv1')
        self.network['conv1_2'] = ConvLayer(self.network['conv1_1'], 64, 3, name='block1_conv2')
        self.network['pool1'] = PoolLayer(self.network['conv1_2'], 2, name='block1_pool')
        self.network['conv2_1'] = ConvLayer(self.network['pool1'], 128, 3, name='block2_conv1')
        self.network['conv2_2'] = ConvLayer(self.network['conv2_1'], 128, 3, name='block2_conv2')
        self.network['pool2'] = PoolLayer(self.network['conv2_2'], 2, name='block2_pool')
        self.network['conv3_1'] = ConvLayer(self.network['pool2'], 256, 3, name='block3_conv1')
        self.network['conv3_2'] = ConvLayer(self.network['conv3_1'], 256, 3, name='block3_conv2')
        self.network['conv3_3'] = ConvLayer(self.network['conv3_2'], 256, 3, name='block3_conv3')
        self.network['conv3_4'] = ConvLayer(self.network['conv3_3'], 256, 3, name='block3_conv4')
        self.network['pool3'] = PoolLayer(self.network['conv3_4'], 2, name='block3_pool')
        self.network['conv4_1'] = ConvLayer(self.network['pool3'], 512, 3, name='block4_conv1')
        self.network['conv4_2'] = ConvLayer(self.network['conv4_1'], 512, 3, name='block4_conv2')
        self.network['conv4_3'] = ConvLayer(self.network['conv4_2'], 512, 3, name='block4_conv3')
        self.network['conv4_4'] = ConvLayer(self.network['conv4_3'], 512, 3, name='block4_conv4')
        self.network['pool4'] = PoolLayer(self.network['conv4_4'], 2, name='block4_pool')
        self.network['conv5_1'] = ConvLayer(self.network['pool4'], 512, 3, name='block5_conv1')
        self.network['conv5_2'] = ConvLayer(self.network['conv5_1'], 512, 3, name='block5_conv2')
        self.network['conv5_3'] = ConvLayer(self.network['conv5_2'], 512, 3, name='block5_conv3')
        self.network['conv5_4'] = ConvLayer(self.network['conv5_3'], 512, 3, name='block5_conv4')

    def setup_discriminator(self):

        def batch_norm(input_layer):
            return BatchNormalization()(input_layer)

        def ConcatLayer(input_layer_arr):
            return Add()(input_layer_arr)

        c = args.discriminator_size
        self.make_layer('disc1.1', batch_norm(self.network['conv1_2']), 1 * c, filter_size=(5, 5), stride=(2, 2),
                        pad=(2, 2))
        self.make_layer('disc1.2', self.last_layer(), 1 * c, filter_size=(5, 5), stride=(2, 2), pad=(2, 2))
        self.make_layer('disc2', batch_norm(self.network['conv2_2']), 2 * c, filter_size=(5, 5), stride=(2, 2),
                        pad=(2, 2))
        self.make_layer('disc3', batch_norm(self.network['conv3_2']), 3 * c, filter_size=(3, 3), stride=(1, 1),
                        pad=(1, 1))
        hypercolumn = ConcatLayer([self.network['disc1.2>'], self.network['disc2>'], self.network['disc3>']])
        self.make_layer('disc4', hypercolumn, 4 * c, filter_size=(1, 1), stride=(1, 1), pad=(0, 0))
        self.make_layer('disc5', self.last_layer(), 3 * c, filter_size=(3, 3), stride=(2, 2))
        self.make_layer('disc6', self.last_layer(), 2 * c, filter_size=(1, 1), stride=(1, 1), pad=(0, 0))
        # self.network['disc'] = batch_norm(ConvLayer(self.last_layer(), 1, filter_size=(1, 1),
        #                                             nonlinearity=lasagne.nonlinearities.linear))
        self.network['disc'] = batch_norm(Conv2D(filters=1, kernel_size=(1, 1))(self.last_layer()))
        # ------------------------------------------------------------------------------------------------------------------
        # Input / Output
        # ------------------------------------------------------------------------------------------------------------------

    def load_perceptual(self):
        # load weights from vgg19 to our vgg19 layers
        vgg_map = {
            'block1_conv1': 'conv1_1',
            'block1_conv2': 'conv1_2',
            'block1_pool': 'pool1',
            'block2_conv1': 'conv2_1',
            'block2_conv2': 'conv2_2',
            'block2_pool': 'pool2',
            'block3_conv1': 'conv3_1',
            'block3_conv2': 'conv3_2',
            'block3_conv3': 'conv3_3',
            'block3_conv4': 'conv3_4',
            'block3_pool': 'pool3',
            'block4_conv1': 'conv4_1',
            'block4_conv2': 'conv4_2',
            'block4_conv3': 'conv4_3',
            'block4_conv4': 'conv4_4',
            'block4_pool': 'pool4',
            'block5_conv1': 'conv5_1',
            'block5_conv2': 'conv5_2',
            'block5_conv3': 'conv5_3',
            'block5_conv4': 'conv5_4'
        }
        # set the weights of the vgg19 layers
        vgg19 = VGG19(input_tensor=self.network['percept'], include_top=False)
        for layer in vgg19.layers:
            if layer.name in vgg_map:
                self.network[vgg_map[layer.name]].set_weights(layer.get_weights())
                self.network[vgg_map[layer.name]].trainable = False

    def list_generator_layers(self):
        model = Model(self.network['img'], self.network['out'])
        for layer in model.layers:
            name = layer.name
            yield (name, layer)

    def save_generator(self):
        self.model_generator.save(self.get_filename(absolute=True))
        config = {k: getattr(args, k) for k in ['generator_blocks', 'generator_residual', 'generator_filters'] + \
                  ['generator_upscale', 'generator_downscale']}

        pickle.dump(config, self.get_config_filename(absolute=True), 'wb')
        print('  - Saved model as `{}` after training.'.format(self.get_filename()))
        print('  - Saved config as `{}` after training.'.format(self.get_config_filename()))

    def load_config(self):
        if not os.path.exists(self.get_config_filename(absolute=True)):
            if args.train:
                return {}
            error("Config file with pre-trained convolution layers not found. Download it here...",
                  "https://github.com/jaretburkett/neural-enlarge/releases/download/v%s/%s" % (
                      __version__, self.get_config_filename()))
        print('  - Loaded config file `{}`.'.format(self.get_config_filename()))
        return pickle.load(self.get_config_filename(absolute=True), 'rb')

    def load_generator(self):
        self.model_generator = Model(self.network['seed'], self.network['out'])
        if os.path.exists(self.get_filename(absolute=True)):
            model.load_weights(self.get_filename(absolute=True))
        else:
            if not args.train:
                error("Model file with pre-trained convolution layers not found. Download it here...",
                      "https://github.com/jaretburkett/neural-enlarge/releases/download/v%s/%s" % (
                          __version__, self.get_filename()))

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

    # def compile(self, params):
    #     # Helper function for rendering test images during training, or standalone inference mode.
    #     model = Model(self.network['seed'], self.network['out'])
    #     # todo change optimizer
    #     adam = optimizers.Adam(lr=1e-3)
    #     model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
    #     model.summary()
    #
    #     # load weights
    #     self.model = model
    #     self.load_weights(params)
    #     self.save()

        # ------------------------------------------------------------------------------------------------------------------
        # Training & Loss Functions
        # ------------------------------------------------------------------------------------------------------------------

    def loss_perceptual(self, y_true, y_pred):
        return losses.mse(y_true=y_true, y_pred=y_pred)

    # def loss_total_variation(self, x):
    #     return T.mean(
    #         ((x[:, :, :-1, :-1] - x[:, :, 1:, :-1]) ** 2 + (x[:, :, :-1, :-1] - x[:, :, :-1, 1:]) ** 2) ** 1.25)

    def total_variation_loss(self, y_true, y_pred):
        if K.image_data_format() == 'channels_first':
            a = K.square(y_true[:, :, :-1, :-1] - y_pred[:, :, 1:, :-1])
            b = K.square(y_true[:, :, :-1, :-1] - y_pred[:, :, :-1, 1:])
        else:
            a = K.square(y_true[:, :-1, :-1, :] - y_pred[:, 1:, :-1, :])
            b = K.square(y_true[:, :-1, :-1, :] - y_pred[:, :-1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))

    def loss_adversarial(self, d):
        return T.mean(1.0 - T.nnet.softminus(d[args.batch_size:]))

    def loss_discriminator(self, d):
        return T.mean(T.nnet.softminus(d[args.batch_size:]) - T.nnet.softplus(d[:args.batch_size]))

    def predict(self, img_arr):
        scald = []
        repro = []
        for x in img_arr:
            scald.append(x)
            # training_seeds = np.transpose(x, (2, 1, 0))  # This is correct, not sure why it is flipped
            training_seeds = np.transpose(x, (1, 2, 0))
            processed = self.model_generator.predict(np.array([training_seeds]))
            repro.append(np.transpose(processed[0], (2, 0, 1)))

        return scald, repro

    def compile(self):
        # Helper function for rendering test images during training, or standalone inference mode.
        input_tensor = T.tensor4()
        seed_tensor = T.tensor4()
        input_layers = {self.network['img']: input_tensor, self.network['seed']: seed_tensor}
        output = lasagne.layers.get_output([self.network[k] for k in ['seed', 'out']], input_layers,
                                           deterministic=True)
        # self.predict = theano.function([seed_tensor], output)

        if not args.train:
            return

        output_layers = [self.network['out'], self.network[args.perceptual_layer], self.network['disc']]
        gen_out, percept_out, disc_out = lasagne.layers.get_output(output_layers, input_layers, deterministic=False)

        # Generator loss function, parameters and updates.
        self.gen_lr = 0.0
        self.adversary_weight = 0.0
        gen_losses = [self.loss_perceptual(percept_out) * args.perceptual_weight,
                      self.loss_total_variation(gen_out) * args.smoothness_weight,
                      self.loss_adversarial(disc_out) * self.adversary_weight]
        gen_params = lasagne.layers.get_all_params(self.network['out'], trainable=True)
        print('  - {} tensors learned for generator.'.format(len(gen_params)))
        gen_updates = lasagne.updates.adam(sum(gen_losses, 0.0), gen_params, learning_rate=self.gen_lr)

        # Discriminator loss function, parameters and updates.
        self.disc_lr = theano.shared(np.array(0.0, dtype=theano.config.floatX))
        disc_losses = [self.loss_discriminator(disc_out)]
        disc_params = list(itertools.chain(*[l.get_params() for k, l in self.network.items() if 'disc' in k]))
        print('  - {} tensors learned for discriminator.'.format(len(disc_params)))
        grads = [g.clip(-5.0, +5.0) for g in T.grad(sum(disc_losses, 0.0), disc_params)]

        disc_updates = lasagne.updates.adam(grads, disc_params, learning_rate=self.disc_lr)

        # Combined Theano function for updating both generator and discriminator at the same time.
        updates = collections.OrderedDict(list(gen_updates.items()) + list(disc_updates.items()))
        self.fit = theano.function([input_tensor, seed_tensor], gen_losses + [disc_out.mean(axis=(1, 2, 3))],
                                   updates=updates)

        self.model_discriminator(self.network['seed'], self.network['disc'])
        self.model_perceptual = Model(self.network['percept_input'], self.network[args.perceptual_layer])


    def fit(self, images, seeds):
        self.history = self.model.train_on_batch(seeds, images)
        return self.history

    def set_discriminator_trainable(self, is_trainable):
        self.model_discriminator.trainable = is_trainable
        # set all layers to trainable state
        for x in self.model_discriminator.layers:
            x.trainable = is_trainable

    def fit(self, images, seeds, images2=None, seeds2=None):
        # images are hr, seeds are lr
        batch_size, iw, ih, ic = seeds.shape

        # ----------------------
        #  Train Discriminator
        # ----------------------
        self.set_discriminator_trainable(True)
        # Sample images and their conditioning counterparts
        imgs_hr, imgs_lr = images, seeds

        # From low res. image generate high res. version
        fake_hr = self.model.predict(imgs_lr)

        valid = np.ones((batch_size,))
        fake = np.zeros((batch_size,))

        # Train the discriminators (original images = real / generated = Fake)
        d_loss_real = self.model_discriminator.train_on_batch(imgs_hr, valid)
        d_loss_fake = self.model_discriminator.train_on_batch(fake_hr, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # d_loss = np.add(d_loss_real, d_loss_fake)

        # ----------------------------------
        #  Train Generator (model) using GAN
        # ----------------------------------

        # set the descriminator as not trainable for the gan
        self.set_discriminator_trainable(False)

        # Extract ground truth image features using pre-trained VGG19 model
        image_features = self.model_perceptual.predict(imgs_hr)

        # Train the generators
        g_loss = self.model_perceptual.train_on_batch(imgs_lr, image_features)

        # only send back stats for GAN combined
        self.history = (g_loss[0], d_loss[0])
        return self.history


# %%
if __name__ == "__main__":
    with tf.device('/CPU:0'):
        model = LasagneModel()
