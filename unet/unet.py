"""TF-UNet written by @juniorxsound <https://orfleisher.com>"""

# Dependencies
from tensorflow.keras.layers import MaxPooling2D, Conv2D, concatenate, Dropout, UpSampling2D  # pylint: disable=import-error
from tensorflow.keras.models import Model  # pylint: disable=import-error


class UNet(Model):
    """A UNet model class for Tensorflow 2.0 using Keras"""

    def __init__(self, name="Unet",
                 activation="relu",
                 padding="same",
                 initializer="he_normal",
                 dropout=0.5):
        """Instance a new UNet model class

        Keyword Arguments:
            name {str} -- The name of the model (default: {"Unet"})
            activation {str} -- Activation function type (default: {"relu"})
            padding {str} -- Kernel padding type (default: {"same"})
            initializer {str} -- Kernel initialization type (default: {"he_normal"})
            dropout {float} -- Dropout rate for droupout layers (default: {0.5})
        """
        super(UNet, self).__init__(name=name)

        self.__init_downsample_block(activation, padding, initializer, dropout)
        self.__init_upsample_block(activation, padding, initializer)

    def call(self, inputs):
        """Forward pass for the UNet

        Arguments:
            inputs {tensorflow.keras.layers.Input} -- The inputs to the network

        Returns:
            tensorflow.keras.layers.Output -- The output of te last layer
        """

        # Downsample blocks
        output = self.__conv1_a(inputs)
        conv_1 = self.__conv1_b(output)  # We store this in conv_1 for upsampling
        output = self.__pool1(conv_1)
        output = self.__conv2_a(output)
        conv_2 = self.__conv2_b(output)  # We store this in conv_3 for upsampling
        output = self.__pool2(conv_2)
        output = self.__conv3_a(output)
        conv_3 = self.__conv3_b(output)  # We store this in conv_3 for upsampling
        output = self.__pool3(conv_3)
        output = self.__conv4_a(output)
        output = self.__conv4_b(output)
        drop_4 = self.__drop4(output)  # We store this in drop_4 for upsampling
        output = self.__pool4(drop_4)
        output = self.__conv5_a(output)
        output = self.__conv5_b(output)
        output = self._drop5(output)

        # Upsample blocks
        output = self.__up6_a(output)
        output = self.__up6_b(output)
        output = concatenate([drop_4, output], axis=3)
        output = self.__conv6_a(output)
        output = self.__conv6_b(output)
        output = self.__up7_a(output)
        output = self.__up7_b(output)
        output = concatenate([conv_3, output], axis=3)
        output = self.__conv7_a(output)
        output = self.__conv7_b(output)
        output = self.__up8_a(output)
        output = self.__up8_b(output)
        output = concatenate([conv_2, output], axis=3)
        output = self.__conv8_a(output)
        output = self.__conv8_b(output)
        output = self.__up9_a(output)
        output = self.__up9_b(output)
        output = concatenate([conv_1, output], axis=3)
        output = self.__conv9_a(output)
        output = self.__conv9_b(output)
        output = self.__conv9_c(output)

        return self.__conv10(output)

    def __init_downsample_block(self,
                                activation,
                                padding,
                                initializer,
                                dropout):
        """Creates the downsample conv blocks

        Arguments:
            activation {str} -- The activation function type
            padding {str} -- The kernel padding type
            initializer {str} -- Kernel initialization type
            dropout {float} -- Dropout rate for dropout layers
        """
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
                              initializer):
        """Creates the upsample conv and deconv blocks

        Arguments:
            activation {str} -- The activation function type
            padding {str} -- The kernel padding type
            initializer {str} -- Kernel initialization type
            dropout {float} -- Dropout rate for dropout layers
        """
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
    UNET = UNet()
    UNET.build((None, 480, 640, 1))
    UNET.summary()
