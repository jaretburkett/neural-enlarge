from keras.layers import Layer
import tensorflow as tf
import keras.backend as K
import itertools

class SubPixelUpscaling(Layer):

    def __init__(self, r, channels, **kwargs):
        super(SubPixelUpscaling, self).__init__(**kwargs)

        self.r = r
        self.channels = channels

    def build(self, input_shape):
        pass

    def _phase_shift_th(self, input, scale, channels):
        import theano.tensor as T
        b, k, row, col = input.shape
        output_shape = (b, channels, row * scale, col * scale)
        out = T.zeros(output_shape)
        r = scale
        for y, x in itertools.product(range(scale), repeat=2):
            out = T.inc_subtensor(out[:, :, y::r, x::r], input[:, r * y + x:: r * r, :, :])
        return out

    def _phase_shift(self, I, r, batch_size):
        # Helper function with main phase shift operation
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (batch_size, a, b, r, r))
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
        X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
        return tf.reshape(X, (batch_size, a * r, b * r, 1))

    # NOTE:test without batchsize
    def _phase_shift_test(self, I, r):
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (1, a, b, r, r))
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, b, a*r, r
        X = tf.split(X, b, 0)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, a*r, b*r
        return tf.reshape(X, (1, a * r, b * r, 1))

    def call(self, x, mask=None):
        if K.backend() == "theano":
            # y = depth_to_scale_th(x, self.r, self.channels)
            y = self._phase_shift_th(x, self.r, self.channels)
        else:
            batch_size = K.int_shape(x)[0]
            # return self.PS(x, self.r, batch_size)

            Xc = tf.split(x, self.channels, 3)
            if batch_size:
                X = tf.concat([self._phase_shift(x, self.r, batch_size) for x in Xc], 3)  # Do the concat RGB
            else:
                X = tf.concat([self._phase_shift_test(x, self.r) for x in Xc], 3)  # Do the concat RGB
            return X
        return y

    def compute_output_shape(self, input_shape):
        if K.image_dim_ordering() == "th":
            b, k, r, c = input_shape
            return b, self.channels, r * self.r, c * self.r
        else:
            b, r, c, k = input_shape
            return b, r * self.r, c * self.r, self.channels

    def get_config(self):
        config = {'r': self.r, 'channels': self.channels}
        base_config = super(SubPixelUpscaling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
