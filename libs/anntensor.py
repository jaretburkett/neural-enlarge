import os
import sys
import bz2
import pickle
import itertools
import collections
import numpy as np
from libs.args import args
from libs.version import __version__
from libs.console import error, extend, ansi

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from keras import backend as K


# Support ansi colors in Windows too.
if sys.platform == 'win32':
    import colorama

# Load the underlying deep learning libraries based on the device specified.  If you specify THEANO_FLAGS manually,
# the code assumes you know what you are doing and they are not overriden!
os.environ.setdefault('THEANO_FLAGS', 'floatX=float32,device={},force_device=True,allow_gc=True,' \
                                      'print_active_device=False'.format(args.device))

# Numeric Computing (GPU)
import theano, theano.tensor as T

T.nnet.softminus = lambda x: x - T.nnet.softplus(x)

# Deep Learning Framework
import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer, Deconv2DLayer as DeconvLayer, Pool2DLayer as PoolLayer
from lasagne.layers import InputLayer, ConcatLayer, ElemwiseSumLayer, batch_norm

import keras
from keras import backend as K
from keras.layers import Conv2D, Conv2DTranspose as DeconvLayer, MaxPool2D, MaxPooling2D
from keras.layers import InputLayer, Concatenate, BatchNormalization, Add, Lambda, Activation
from keras.layers import PReLU, ZeroPadding2D, Input
from keras.utils.data_utils import get_file
from keras.applications import vgg19


class SubpixelReshuffleLayer(lasagne.layers.Layer):
    """Based on the code by ajbrock: https://github.com/ajbrock/Neural-Photo-Editor/
    """

    def __init__(self, incoming, channels, upscale, **kwargs):
        super(SubpixelReshuffleLayer, self).__init__(incoming, **kwargs)
        self.upscale = upscale
        self.channels = channels

    def get_output_shape_for(self, input_shape):
        def up(d): return self.upscale * d if d else d

        return input_shape[0], self.channels, up(input_shape[2]), up(input_shape[3])

    def get_output_for(self, input, deterministic=False, **kwargs):
        out, r = K.zeros(self.get_output_shape_for(input.shape)), self.upscale
        for y, x in itertools.product(range(r), repeat=2):
            out = T.inc_subtensor(out[:, :, y::r, x::r], input[:, r * y + x::r * r, :, :])
        return out


class Subpixel(Conv2D):
    """Based on the code by tetrachrome: https://github.com/tetrachrome/subpixel/
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 upscale,
                 padding='valid',
                 data_format=None,
                 strides=(1,1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Subpixel, self).__init__(
            # r=upscale,
            filters=upscale*upscale*filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.r = upscale

    def _phase_shift(self, I):
        r = self.r
        bsize, a, b, c = I.get_shape().as_list()
        bsize = K.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
        X = K.reshape(I, [bsize, a, b, c/(r*r),r, r]) # bsize, a, b, c/(r*r), r, r
        X = K.permute_dimensions(X, (0, 1, 2, 5, 4, 3))  # bsize, a, b, r, r, c/(r*r)
        #Keras backend does not support tf.split, so in future versions this could be nicer
        X = [X[:,i,:,:,:,:] for i in range(a)] # a, [bsize, b, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, b, a*r, r, c/(r*r)
        X = [X[:,i,:,:,:] for i in range(b)] # b, [bsize, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, a*r, b*r, c/(r*r)
        return X

    def call(self, inputs):
        return self._phase_shift(super(Subpixel, self).call(inputs))

    def compute_output_shape(self, input_shape):
        unshifted = super(Subpixel, self).compute_output_shape(input_shape)
        return (unshifted[0], self.r*unshifted[1], self.r*unshifted[2], unshifted[3]/(self.r*self.r))

    def get_config(self):
        config = super(Conv2D, self).get_config()
        config.pop('rank')
        config.pop('dilation_rate')
        config['filters']/=self.r*self.r
        config['r'] = self.r
        return config


class Model(object):

    def __init__(self):
        self.network = collections.OrderedDict()
        self.network['img'] = InputLayer((None, 3, None, None))
        self.network['seed'] = InputLayer((None, 3, None, None))

        # import model
        self.model = None

        config, params = self.load_model()
        self.setup_generator(self.last_layer(), config)

        if args.train:
            concatenated = Concatenate([self.network['img'], self.network['out']], axis=0)
            self.setup_perceptual(concatenated)
            self.load_perceptual()
            self.setup_discriminator()
        self.load_generator(params)
        self.compile()

    # ------------------------------------------------------------------------------------------------------------------
    # Network Configuration
    # ------------------------------------------------------------------------------------------------------------------

    def last_layer(self):
        return list(self.network.values())[-1]

    def make_layer(self, name, input, units, filter_size=(3, 3), stride=(1, 1), pad=(1, 1), alpha=0.25):
        # conv = ConvLayer(input, units, filter_size, stride=stride, pad=pad, nonlinearity=None)
        # prelu = lasagne.layers.ParametricRectifierLayer(conv, alpha=lasagne.init.Constant(alpha))
        # self.network[name + 'x'] = conv
        # self.network[name + '>'] = prelu

        #keras
        padding = ZeroPadding2D(padding=pad, name=name + '_')(input)
        conv = Conv2D(filters=units, kernel_size=filter_size, strides=stride, padding='valid', name=name + 'x')(padding)
        prelu = PReLU(alpha_initializer=keras.initializers.Constant(alpha), name=name + '>')(conv)
        self.network[name + '_'] = padding
        self.network[name + 'x'] = conv
        self.network[name + '>'] = prelu
        return prelu

    def make_block(self, name, input, units):
        self.make_layer(name + '-A', input, units, alpha=0.1)
        # self.make_layer(name+'-B', self.last_layer(), units, alpha=1.0)
        # return ElemwiseSumLayer([input, self.last_layer()]) if args.generator_residual else self.last_layer()
        #keras
        return Add()([input, self.last_layer()]) if args.generator_residual else self.last_layer()

    def setup_generator(self, input, config):
        for k, v in config.items(): setattr(args, k, v)
        args.zoom = 2 ** (args.generator_upscale - args.generator_downscale)

        units_iter = extend(args.generator_filters)
        units = next(units_iter)
        self.make_layer('iter.0', input, units, filter_size=(7, 7), pad=(3, 3))

        for i in range(0, args.generator_downscale):
            self.make_layer('downscale%i' % i, self.last_layer(), next(units_iter), filter_size=(4, 4), stride=(2, 2))

        units = next(units_iter)
        for i in range(0, args.generator_blocks):
            self.make_block('iter.%i' % (i + 1), self.last_layer(), units)

        for i in range(0, args.generator_upscale):
            u = next(units_iter)
            self.make_layer('upscale%i.2' % i, self.last_layer(), u * 4)
            # self.network['upscale%i.1' % i] = SubpixelReshuffleLayer(self.last_layer(), u, 2)
            # todo not sure which kernel size to use
            self.network['upscale%i.1' % i] = Subpixel(filters=u, upscale=2, kernel_size=(3, 3))(self.last_layer())

        # self.network['out'] = ConvLayer(self.last_layer(), 3, filter_size=(7, 7), pad=(3, 3), nonlinearity=None)
        padding = ZeroPadding2D(padding=(3, 3), name='out_')(self.last_layer())
        self.network['out'] = Conv2D(filters=3, kernel_size=(7, 7))(padding)

    def setup_perceptual(self, input):
        # keras
        # todo not sure if this percept part is right
        offset = np.array([103.939, 116.779, 123.680], dtype=np.float32).reshape((1, 3, 1, 1))
        self.network['percept'] = Lambda(lambda x: ((x + 0.5) * 255.0) - offset)(input)
        self.network['mse'] = self.network['percept']

        # Block 1
        self.network['conv1_1'] = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(self.network['percept'])
        self.network['conv1_2'] = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(self.network['conv1_1'])
        self.network['pool1'] = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(self.network['conv1_2'])

        # Block 2
        self.network['conv2_1'] = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(self.network['pool1'])
        self.network['conv2_2'] = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(self.network['conv2_1'])
        self.network['pool2'] = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(self.network['conv2_2'])

        # Block 3
        self.network['conv3_1'] = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(self.network['pool2'])
        self.network['conv3_2'] = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(self.network['conv3_1'])
        self.network['conv3_3'] = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(self.network['conv3_2'])
        self.network['conv3_4'] = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(self.network['conv3_3'])
        self.network['pool3'] = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(self.network['conv3_4'])

        # Block 4
        self.network['conv4_1'] = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(self.network['pool3'])
        self.network['conv4_2'] = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(self.network['conv4_1'])
        self.network['conv4_3'] = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(self.network['conv4_2'])
        self.network['conv4_4'] = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(self.network['conv4_3'])
        self.network['pool4'] = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(self.network['conv4_4'])

        # Block 5
        self.network['conv5_1'] = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(self.network['pool4'])
        self.network['conv5_2'] = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(self.network['conv5_1'])
        self.network['conv5_3'] = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(self.network['conv5_2'])
        self.network['conv5_4'] = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(self.network['conv5_3'])

        # todo this wasnt included in lasagne version
        # self.network['pool5'] = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(self.network['conv5_4'])

    def setup_discriminator(self):
        #keras
        c = args.discriminator_size
        self.make_layer('disc1.1', BatchNormalization()(self.network['conv1_2']), 1 * c, filter_size=(5, 5), stride=(2, 2),
                        pad=(2, 2))
        self.make_layer('disc1.2', self.last_layer(), 1 * c, filter_size=(5, 5), stride=(2, 2), pad=(2, 2))
        self.make_layer('disc2', BatchNormalization()(self.network['conv2_2']), 2 * c, filter_size=(5, 5), stride=(2, 2),
                        pad=(2, 2))
        self.make_layer('disc3', BatchNormalization()(self.network['conv3_2']), 3 * c, filter_size=(3, 3), stride=(1, 1),
                        pad=(1, 1))
        hypercolumn = Concatenate([self.network['disc1.2>'], self.network['disc2>'], self.network['disc3>']])
        self.make_layer('disc4', hypercolumn, 4 * c, filter_size=(1, 1), stride=(1, 1), pad=(0, 0))
        self.make_layer('disc5', self.last_layer(), 3 * c, filter_size=(3, 3), stride=(2, 2))
        self.make_layer('disc6', self.last_layer(), 2 * c, filter_size=(1, 1), stride=(1, 1), pad=(0, 0))
        # todo verify axis of batch normalization
        self.network['disc'] = BatchNormalization(name='disc')(Conv2D(filters=1, kernel_size=(1, 1), activation='linear')(self.last_layer()))

    # ------------------------------------------------------------------------------------------------------------------
    # Input / Output
    # ------------------------------------------------------------------------------------------------------------------

    def load_perceptual(self):
        """Load the weights from pretrained vgg19 network and load them in our model
        """
        from keras.applications import vgg19
        vgg19_model = vgg19.VGG19(False)
        vgg_layer = dict([(layer.name, layer) for layer in vgg19_model.layers])
        ne_layer = dict([(layer.name, layer) for layer in self.model.layers])

        weights_to_copy = [
            'block1_conv1', 'block1_conv2', 'block1_pool',
            'block2_conv1', 'block2_conv2', 'block2_pool',
            'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_conv4', 'block3_pool',
            'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_conv4', 'block4_pool',
            'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_conv4'
        ]
        for lname in weights_to_copy:
            ne_layer[lname].set_weights(vgg_layer[lname].get_weights())

    def list_generator_layers(self):
        for l in lasagne.layers.get_all_layers(self.network['out'], treat_as_input=[self.network['img']]):
            if not l.get_params(): continue
            name = list(self.network.keys())[list(self.network.values()).index(l)]
            yield (name, l)

    def get_filename(self, absolute=False):
        filename = 'models/ne%ix-%s-%s-%s.h5' % (args.zoom, args.type, args.model, __version__)
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', filename)) if absolute else filename

    def save_generator(self):
        def cast(p): return p.get_value().astype(np.float16)

        params = {k: [cast(p) for p in l.get_params()] for (k, l) in self.list_generator_layers()}
        config = {k: getattr(args, k) for k in ['generator_blocks', 'generator_residual', 'generator_filters'] + \
                  ['generator_upscale', 'generator_downscale']}

        pickle.dump((config, params), bz2.open(self.get_filename(absolute=True), 'wb'))
        print('  - Saved model as `{}` after training.'.format(self.get_filename()))

    def load_keras_model(self):
        if not os.path.exists(self.get_filename(absolute=True)):
            if args.train:
                return None
            else:
                error("Model file with pre-trained convolution layers not found. Download it here...",
                  "https://github.com/jaretburkett/neural-enlarge/releases/download/v%s/%s" % (
                      __version__, self.get_filename()))
        else:
            return keras.models.load_model(self.get_filename(absolute=True))


    # def load_model(self):
    #     if not os.path.exists(self.get_filename(absolute=True)):
    #         if args.train:
    #             return {}, {}
    #         error("Model file with pre-trained convolution layers not found. Download it here...",
    #               "https://github.com/jaretburkett/neural-enlarge/releases/download/v%s/%s" % (
    #               __version__, self.get_filename()))
    #     print('  - Loaded file `{}` with trained model.'.format(self.get_filename()))
    #     return pickle.load(bz2.open(self.get_filename(absolute=True), 'rb'))

    def load_generator(self, params):
        if len(params) == 0:
            return
        for k, l in self.list_generator_layers():
            assert k in params, "Couldn't find layer `%s` in loaded model.'" % k
            assert len(l.get_params()) == len(params[k]), "Mismatch in types of layers."
            for p, v in zip(l.get_params(), params[k]):
                assert v.shape == p.get_value().shape, "Mismatch in number of parameters for layer {}.".format(k)
                p.set_value(v.astype(np.float32))

    # ------------------------------------------------------------------------------------------------------------------
    # Training & Loss Functions
    # ------------------------------------------------------------------------------------------------------------------

    def loss_perceptual(self, p):
        return lasagne.objectives.squared_error(p[:args.batch_size], p[args.batch_size:]).mean()

    def loss_total_variation(self, x):
        return T.mean(
            ((x[:, :, :-1, :-1] - x[:, :, 1:, :-1]) ** 2 + (x[:, :, :-1, :-1] - x[:, :, :-1, 1:]) ** 2) ** 1.25)

    def loss_adversarial(self, d):
        return T.mean(1.0 - T.nnet.softminus(d[args.batch_size:]))

    def loss_discriminator(self, d):
        return T.mean(T.nnet.softminus(d[args.batch_size:]) - T.nnet.softplus(d[:args.batch_size]))

    def compile(self):
        # Helper function for rendering test images during training, or standalone inference mode.
        # input_tensor, seed_tensor = T.tensor4(), T.tensor4()
        input_tensor, seed_tensor = InputLayer((None, 3, None, None)), InputLayer((None, 3, None, None))
        input_layers = {self.network['img']: input_tensor, self.network['seed']: seed_tensor}
        output = lasagne.layers.get_output([self.network[k] for k in ['seed', 'out']], input_layers, deterministic=True)
        # self.predict = theano.function([seed_tensor], output)
        self.predict = theano.function([seed_tensor], output)

        if not args.train: return

        output_layers = [self.network['out'], self.network[args.perceptual_layer], self.network['disc']]
        gen_out, percept_out, disc_out = lasagne.layers.get_output(output_layers, input_layers, deterministic=False)

        # Generator loss function, parameters and updates.
        self.gen_lr = theano.shared(np.array(0.0, dtype=theano.config.floatX))
        self.adversary_weight = theano.shared(np.array(0.0, dtype=theano.config.floatX))
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


print('{}  - Using the device `{}` for neural computation.{}\n'.format(ansi.CYAN, theano.config.device, ansi.ENDC))