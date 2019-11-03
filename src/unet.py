"""TF-UNet written by @juniorxsound <https://orfleisher.com>"""

# Dependencies
from tensorflow.keras.layers import MaxPooling2D, Conv2D, concatenate, Dropout, UpSampling2D
from tensorflow.keras.models import Model


class UNet(Model):
    """A UNet model class for Tensorflow 2.0 using Keras"""

    def __init__(self, name="Unet",
                 activation="relu",
                 padding="same",
                 initializer="he_normal",
                 dropout=0.5):
        super(UNet, self).__init__(name=name)

        self.__init_downsample_block(activation, padding, initializer, dropout)
        self.__init_upsample_block(activation, padding, initializer, dropout)

    def call(self, inputs):
        """Downsample forward pass"""
        x = self.__conv1_a(inputs)
        conv_1 = self.__conv1_b(x)  # We store this in conv_1 for upsampling
        x = self.__pool1(conv_1)
        x = self.__conv2_a(x)
        conv_2 = self.__conv2_b(x)  # We store this in conv_3 for upsampling
        x = self.__pool2(conv_2)
        x = self.__conv3_a(x)
        conv_3 = self.__conv3_b(x)  # We store this in conv_3 for upsampling
        x = self.__pool3(conv_3)
        x = self.__conv4_a(x)
        x = self.__conv4_b(x)
        drop_4 = self.__drop4(x)  # We store this in drop_4 for upsampling
        x = self.__pool4(drop_4)
        x = self.__conv5_a(x)
        x = self.__conv5_b(x)
        x = self._drop5(x)

        """Upsample forward pass"""
        x = self.__up6_a(x)
        x = self.__up6_b(x)
        x = concatenate([drop_4, x], axis=3)
        x = self.__conv6_a(x)
        x = self.__conv6_b(x)
        x = self.__up7_a(x)
        x = self.__up7_b(x)
        x = concatenate([conv_3, x], axis=3)
        x = self.__conv7_a(x)
        x = self.__conv7_b(x)
        x = self.__up8_a(x)
        x = self.__up8_b(x)
        x = concatenate([conv_2, x], axis=3)
        x = self.__conv8_a(x)
        x = self.__conv8_b(x)
        x = self.__up9_a(x)
        x = self.__up9_b(x)
        x = concatenate([conv_1, x], axis=3)
        x = self.__conv9_a(x)
        x = self.__conv9_b(x)
        x = self.__conv9_c(x)

        return self.__conv10(x)

    def __init_downsample_block(self,
                                activation,
                                padding,
                                initializer,
                                dropout):
        self.__conv1_a = Conv2D(64, 3, activation=activation, padding=padding,
                                kernel_initializer=initializer)
        self.__conv1_b = Conv2D(64, 3, activation=activation, padding=padding,
                                kernel_initializer=initializer)
        self.__pool1 = MaxPooling2D(pool_size=(2, 2))

        self.__conv2_a = Conv2D(128, 3, activation=activation, padding=padding,
                                kernel_initializer=initializer)
        self.__conv2_b = Conv2D(128, 3, activation=activation, padding=padding,
                                kernel_initializer=initializer)
        self.__pool2 = MaxPooling2D(pool_size=(2, 2))

        self.__conv3_a = Conv2D(256, 3, activation=activation, padding=padding,
                                kernel_initializer=initializer)
        self.__conv3_b = Conv2D(256, 3, activation=activation, padding=padding,
                                kernel_initializer=initializer)
        self.__pool3 = MaxPooling2D(pool_size=(2, 2))

        self.__conv4_a = Conv2D(512, 3, activation=activation, padding=padding,
                                kernel_initializer=initializer)
        self.__conv4_b = Conv2D(512, 3, activation=activation, padding=padding,
                                kernel_initializer=initializer)
        self.__drop4 = Dropout(dropout)
        self.__pool4 = MaxPooling2D(pool_size=(2, 2))

        self.__conv5_a = Conv2D(1024, 3, activation=activation, padding=padding,
                                kernel_initializer=initializer)
        self.__conv5_b = Conv2D(1024, 3, activation=activation, padding=padding,
                                kernel_initializer=initializer)

        self._drop5 = Dropout(dropout)

    def __init_upsample_block(self,
                              activation,
                              padding,
                              initializer,
                              dropout):
        self.__up6_a = UpSampling2D(size=(2, 2))
        self.__up6_b = Conv2D(512, 3, activation=activation, padding=padding,
                              kernel_initializer=initializer)

        self.__conv6_a = Conv2D(512, 3, activation=activation, padding=padding,
                                kernel_initializer=initializer)
        self.__conv6_b = Conv2D(512, 3, activation=activation, padding=padding,
                                kernel_initializer=initializer)

        self.__up7_a = UpSampling2D(size=(2, 2))
        self.__up7_b = Conv2D(256, 3, activation=activation, padding=padding,
                              kernel_initializer=initializer)

        self.__conv7_a = Conv2D(256, 3, activation=activation, padding=padding,
                                kernel_initializer=initializer)
        self.__conv7_b = Conv2D(256, 3, activation=activation, padding=padding,
                                kernel_initializer=initializer)

        self.__up8_a = UpSampling2D(size=(2, 2))
        self.__up8_b = Conv2D(128, 2, activation=activation, padding=padding,
                              kernel_initializer=initializer)

        self.__conv8_a = Conv2D(128, 3, activation=activation, padding=padding,
                                kernel_initializer=initializer)
        self.__conv8_b = Conv2D(128, 3, activation=activation, padding=padding,
                                kernel_initializer=initializer)

        self.__up9_a = UpSampling2D(size=(2, 2))
        self.__up9_b = Conv2D(64, 2, activation=activation, padding=padding,
                              kernel_initializer=initializer)

        self.__conv9_a = Conv2D(64, 3, activation=activation, padding=padding,
                                kernel_initializer=initializer)
        self.__conv9_b = Conv2D(64, 3, activation=activation, padding=padding,
                                kernel_initializer=initializer)
        self.__conv9_c = Conv2D(2, 3, activation=activation, padding=padding,
                                kernel_initializer=initializer)
        self.__conv10 = Conv2D(1, 1, activation=activation)


if __name__ == "__main__":
    unet = UNet()
    unet.build((None, 480, 640, 1))
    unet.summary()
