
import os
import collections
import numpy as np
from libs.args import args
from keras import optimizers
from keras import backend as K
from libs.losses import PSNRLoss
from libs.version import __version__
from libs.layers import SubPixelUpscaling
from keras.layers import Input, Conv2D, Conv2DTranspose, Add, MaxPooling2D
from keras.models import Model as KerasModel, load_model as load_keras_model


class BaseModel(object):

    def __init__(self):
        self.network = collections.OrderedDict()
        self.history = None
        self.model = None
        self.channels = 3

        self.n1 = 50
        self.n2 = 100
        self.n3 = 200

        # self.load_model()

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
        self.history = self.model.train_on_batch(seeds, images)
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
            processed = self.model.predict(np.array([training_seeds]))
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
        test = np.array([test_img])
        layer_outs = functor([test, 1.])
        each_layer = layer_outs
        return layer_names, each_layer