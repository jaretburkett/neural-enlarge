from keras.layers import Layer
import tensorflow as tf
import keras.backend as K

# based on the code from
# https://github.com/WeidiXie/New_Layers-Keras-Tensorflow


class SubpixelReshuffleLayer(Layer):

    def __init__(self,
                 channels,
                 upscale,
                 data_format='channels_last',
                 **kwargs):
        super(SubpixelReshuffleLayer, self).__init__(**kwargs)
        self.upscale = upscale
        self.channels = channels
        self.data_format = data_format

    def call(self, input, **kwargs):

        def dense_interp(x, r, shape):
            # X = tf.reshape(tensor=I, shape=shape)
            # X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
            # X = tf.split(X, input_shape_as_numbers[1], axis=1)  # a, [bsize, b, r, r]
            # X = tf.concat([tf.squeeze(x) for x in X], axis=2)  # bsize, b, a*r, r
            # X = tf.split(X, input_shape_as_numbers[2], axis=1)  # b, [bsize, a*r, r]
            # X = tf.concat([tf.squeeze(x) for x in X], axis=2)  # bsize, a*r, b*r
            # return tf.reshape(X, (bsize, a * r, b * r, 1))

            # x should be of shape : batch_size, w, h, r^2
            bsize, a, b, c = shape
            X = tf.reshape(x, (-1, a, b, r, r))
            X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
            X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
            X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
            X = tf.reshape(X, (-1, b, a * r, r))
            X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
            X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
            return tf.reshape(X, (-1, a * r, b * r, 1))

        inp_shape = input.get_shape()
        output_shape = self.compute_output_shape(inp_shape)
        nb_channel = output_shape[-1]
        r = self.upscale

        if nb_channel > 1:
            interp_shape = [inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[3] // nb_channel]
            Xc = tf.split(input, nb_channel, 3)
            X = tf.concat([dense_interp(x, r, interp_shape) for x in Xc], 3)
        else:
            interp_shape = inp_shape
            X = dense_interp(input, r, interp_shape)
        return X

    def compute_output_shape(self, input_shape):
        def up(d): return self.upscale * d
        if self.data_format == 'channels_last':
            return input_shape[0], up(input_shape[1]), up(input_shape[2]), self.channels
        else:
            return input_shape[0], self.channels, up(input_shape[2]), up(input_shape[3])



import tensorflow as tf
from keras.engine import InputSpec, Layer


# ===================================
#  Subpixel Dense Up-sampling Layer.
# ===================================

def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    X = tf.reshape(I, (-1, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, 1, a)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, [1]) for x in X], 2)  # bsize, b, a*r, r
    X = tf.split(X, 1, b)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, [1]) for x in X], 2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))


def dense_interp(x, r, shape):
    # x should be of shape : batch_size, w, h, r^2
    #
    # # X = tf.reshape(tensor=I, shape=shape)
    # X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    # X = tf.split(X, input_shape_as_numbers[1], axis=1)  # a, [bsize, b, r, r]
    # X = tf.concat([tf.squeeze(x) for x in X], axis=2)  # bsize, b, a*r, r
    # X = tf.split(X, input_shape_as_numbers[2], axis=1)  # b, [bsize, a*r, r]
    # X = tf.concat([tf.squeeze(x) for x in X], axis=2)  # bsize, a*r, b*r
    # return tf.reshape(X, (bsize, a * r, b * r, 1))


    bsize, a, b, c = shape
    X = tf.reshape(x, (-1, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x) for x in X], 2, name='concatA')  # bsize, b, a*r, r
    X = tf.reshape(X, (-1, b, a * r, r))
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x) for x in X], 2, name='concatB')  # bsize, a*r, b*r
    return tf.reshape(X, (-1, a * r, b * r, 1))


# from keras.engine.topology import Layer

class SubpixDenseUP(Layer):
    '''
    This layer is inspired by the paper[1],
    aiming to provide upsampling in a more accurate way.
    Input to this layer is of size: w x h x r^2*c

    Output from this layer is of size : W x H x C
    where W = w * r, H = h * r, C = c
    Intuitively, we want to compensate the information loss with more feature
    channels, depth -> spatial resolution.
    We can reshape the feature channels,
    for example, take 1 x 1 x k^2, we can reshape it to k x k x 1.
    ratio : the ratio you want to upsample for both dimensions (w,h).
    nb_channel : The channels you really want after up-sampling.
    nb_channel = input_shape[-1] / (prod(ratio))
    dim_ordering = 'tf'
    Reference:
    [1] Real-Time Single Image and Video Super-Resolution Using an Efficient
    Sub-Pixel Convolutional Neural Network.
    '''

    def __init__(self, ratio=2, dim_ordering='tf', **kwargs):
        self.ratio = ratio
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        # self.output_dim = output_dim
        # self.input_spec = [InputSpec(ndim=4)]
        super(SubpixDenseUP, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        # Create a trainable weight variable for this layer.
        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_dim,),
        #                               initializer='uniform',
        #                               trainable=True)
        super(SubpixDenseUP, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'tf':
            height = self.ratio * input_shape[1] if input_shape[1] is not None else None
            width = self.ratio * input_shape[2] if input_shape[2] is not None else None
            if input_shape[3] % (self.ratio ** 2) != 0:
                raise Exception('input shape : {}, can not upsample to {}'.format(input_shape,
                                                                                  (input_shape[0], height, width,
                                                                                   input_shape[3] // self.ratio ** 2)))
            else:
                channel = input_shape[3] // (self.ratio ** 2)
            return (input_shape[0],
                    width,
                    height,
                    channel)
        else:
            raise Exception('Only support TF, Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        inp_shape = x.get_shape()
        # X = x
        # inp_shape = x._keras_shape
        output_shape = self.compute_output_shape(inp_shape)
        nb_channel = output_shape[-1]
        r = self.ratio

        # if nb_channel > 1:
        interp_shape = [inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[3] // nb_channel]
        Xc = tf.split(x, nb_channel, 3)
        X = tf.concat([dense_interp(x, r, interp_shape) for x in Xc], 3)
        # else:
        #     interp_shape = inp_shape
        #     X = dense_interp(x, r, interp_shape)
        return X
        # if nb_channel > 1:
        #     Xc = tf.split(x, inp_shape[-1], 3)
        #     X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
        # else:
        #     X = _phase_shift(x, r)
        # return X

    def get_config(self):
        config = {'ratio': self.ratio}
        base_config = super(SubpixDenseUP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




''' Theano Backend function '''
def depth_to_scale_th(input, scale, channels):
    ''' Uses phase shift algorithm [1] to convert channels/depth for spacial resolution '''
    import theano.tensor as T

    b, k, row, col = input.shape
    output_shape = (b, channels, row * scale, col * scale)

    out = T.zeros(output_shape)
    r = scale

    for y, x in itertools.product(range(scale), repeat=2):
        out = T.inc_subtensor(out[:, :, y::r, x::r], input[:, r * y + x :: r * r, :, :])

    return out


''' Tensorflow Backend Function '''
def depth_to_scale_tf(input, scale, channels):
    try:
        import tensorflow as tf
    except ImportError:
        print("Could not import Tensorflow for depth_to_scale operation. Please install Tensorflow or switch to Theano backend")
        exit()

    def _phase_shift(self, I, r):
        # Helper function with main phase shift operation
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (self.batch_size, a, b, r, r))
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
        X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
        return tf.reshape(X, (self.batch_size, a * r, b * r, 1))

    # NOTE:test without batchsize
    def _phase_shift_test(self, I, r):
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (1, a, b, r, r))
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, b, a*r, r
        X = tf.split(X, b, 0)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, a*r, b*r
        return tf.reshape(X, (1, a * r, b * r, 1))

    if channels > 1:
        Xc = tf.split(input, 3)
        X = tf.concat([_phase_shift(x, scale) for x in Xc], 3)
    else:
        X = _phase_shift(input, scale)
    return X

'''
Implementation is incomplete. Use lambda layer for now.
'''

class SubPixelUpscaling(Layer):

    def __init__(self, r, channels, **kwargs):
        super(SubPixelUpscaling, self).__init__(**kwargs)

        self.r = r
        self.channels = channels

    def build(self, input_shape):
        pass

    def _phase_shift(self, I, r, batch_size):
        # Helper function with main phase shift operation
        bsize, a, b, c = I.get_shape().as_list()
        # bsize = tf.shape(I)[0]
        # a = tf.shape(I)[1]
        # b = tf.shape(I)[2]
        # c = tf.shape(I)[3]
        X = tf.reshape(I, (batch_size, a, b, r, r))
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
        X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
        return tf.reshape(X, (batch_size, a * r, b * r, 1))

    # NOTE:test without batchsize
    def _phase_shift_test(self, I, r):
        bsize, a, b, c = I.get_shape().as_list()
        # bsize = tf.shape(I)[0]
        # a = tf.shape(I)[1]
        # b= tf.shape(I)[2]
        # c = tf.shape(I)[3]
        X = tf.reshape(I, (1, a, b, r, r))
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, b, a*r, r
        X = tf.split(X, b, 0)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, a*r, b*r
        return tf.reshape(X, (1, a * r, b * r, 1))

    def PS(self, X, r, batch_size):
        # Main OP that you can arbitrarily use in you tensorflow code
        Xc = tf.split(X, self.channels, 3)
        if batch_size:
            X = tf.concat([self._phase_shift(x, r, batch_size) for x in Xc], 3)  # Do the concat RGB
        else:
            X = tf.concat([self._phase_shift_test(x, r) for x in Xc], 3)  # Do the concat RGB
        return X

    def call(self, x, mask=None):
        if K.backend() == "theano":
            y = depth_to_scale_th(x, self.r, self.channels)
        else:
            batch_size = K.int_shape(x)[0]
            return self.PS(x, self.r, batch_size)
        return y

    def compute_output_shape(self, input_shape):
        if K.image_dim_ordering() == "th":
            b, k, r, c = input_shape
            return (b, self.channels, r * self.r, c * self.r)
        else:
            b, r, c, k = input_shape
            return (b, r * self.r, c * self.r, self.channels)