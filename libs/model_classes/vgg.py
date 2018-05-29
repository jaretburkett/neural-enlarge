
import os
from libs.args import args
from keras import optimizers
from libs.losses import PSNRLoss
from keras.applications.vgg19 import VGG19
from keras.layers import Input, Conv2D, Conv2DTranspose, Add, MaxPooling2D
from keras.models import Model
from libs.model_classes.base import BaseModel



class VGG19b34(BaseModel):

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

        model = Model(init, decoded)
        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])

        model.summary()

        # load weights
        if os.path.exists(self.get_filename(absolute=True)):
            print('Importing weights from file %s' % self.get_filename(absolute=False))
            model.load_weights(self.get_filename(absolute=True), by_name=True)
        self.model = model

        # return model


class VGG19b13(BaseModel):

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

        model = Model(init, decoded)
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
