import tensorflow as tf

__DEBUG__ = True


class UNet():

    '''
    A static utility method to create tf.Variable for weights
    '''
    @staticmethod
    def _create_weights(shape, name='W'):
        return tf.Variable(tf.random.truncated_normal(shape, stddev=0.05), name=name)

    '''
    A static utility method to create tf.Variable for biases
    '''
    @staticmethod
    def _create_biases(size, name='B'):
        return tf.Variable(tf.constant(0.05, shape=[size]), name=name)

    '''
    @TODO document
    '''
    @staticmethod
    def conv(input,
             input_channels,
             output_channels,
             kernel_size,
             layer_stride=[1, 1, 1, 1],
             use_relu=True,
             name='conv'):
        with tf.name_scope(name):
            # We shall define the weights that will be trained using create_weights function.
            weights = UNet._create_weights(
                shape=[output_channels, output_channels, input_channels, kernel_size])

            # We create biases using the _create_weights function. These are also trained.
            biases = UNet._create_biases(kernel_size)

            # Creating the convolutional layer
            layer = tf.nn.conv2d(input=input,
                                 filter=weights,
                                 strides=layer_stride,
                                 padding='SAME')

            # Add the biases to the layer
            layer += biases

            # Max pool the layer
            layer = tf.nn.max_pool2d(layer,
                                     [1, 2, 2, 1],
                                     strides=[1, 2, 2, 1],
                                     padding='SAME')

            # Output layer is fed to Relu which is the activation function for us.
            if use_relu is True:
                layer = tf.nn.relu(layer)

            # Create summaries for TensorBoard
            tf.compat.v1.summary.histogram("weights", weights)
            tf.compat.v1.summary.histogram("biases", biases)
            tf.compat.v1.summary.histogram("activitions", layer)

            # Debug prints
            if (__DEBUG__):
                print('Layer shape: {}'.format(layer.get_shape()))

            return layer

    '''
    @TODO document
    '''
    @staticmethod
    def upconv(input,
               output_shape,
               input_channels,
               output_channels,
               kernel_size,
               layer_stride=[1, 2, 2, 1],
               use_relu=True,
               name='upconv'):
        with tf.name_scope(name):
            # We shall define the weights that will be trained using _create_weights function.
            weights = UNet._create_weights(
                shape=[output_channels, output_channels, input_channels, kernel_size])

            # We create biases using the _create_weights function. These are also trained.
            biases = UNet._create_biases(kernel_size)

            # Creating the convolutional layer
            layer = tf.nn.conv2d_transpose(value=input,
                                           filter=weights,
                                           output_shape=output_shape,
                                           strides=layer_stride,
                                           padding='SAME')

            # Add the biases to the layer
            layer += biases

            # Output layer is fed to Relu which is the activation function for us.
            if use_relu is True:
                layer = tf.nn.relu(layer)

            # Create summaries for TensorBoard
            tf.compat.v1.summary.histogram("weights", weights)
            tf.compat.v1.summary.histogram("biases", biases)
            tf.compat.v1.summary.histogram("activitions", layer)

            # Debug prints
            if (__DEBUG__):
                print('Layer shape: {}'.format(layer.get_shape()))

            return layer


'''
Used for unit testing the creation of each layer ⚠️ should not be used by directly, see train.py for the proper interface
'''
if __name__ == "__main__":

    # If for some reason debug is false set it to true, since this block only runs for testing so we want debug prints
    if __DEBUG__ is False:
        __DEBUG__ = True

    # Create a test input layer
    x = tf.compat.v1.placeholder(
        tf.float32, shape=[None, 180, 320, 3], name="x")

    # Create down conv layer
    print('\nCreating down conv layer (batch_dim, height, width, depth)')
    print('Input shape: {}'.format(x.get_shape()))
    layer = UNet.conv(x, 3, 3, 32)

    # Assert shapes
    assert int(x.get_shape()[1]) / 2 == layer.get_shape()[1]
    assert int(x.get_shape()[2]) / 2 == layer.get_shape()[2]
    print('✅ Down conv reduces height and width by a factor of 2')

    # Create upconv (transposed conv) layer
    print('\nCreating upconv layer (batch_dim, height, width, depth)')
    print('Input shape: {}'.format(x.get_shape()))
    layer = UNet.upconv(x, [-1, 360, 640, 3], 3, 3, 3)

    # Assert shapes
    assert int(x.get_shape()[1]) * 2 == layer.get_shape()[1]
    assert int(x.get_shape()[2]) * 2 == layer.get_shape()[2]
    print('✅ Transposed conv doubles height, width but keeps depth the SAME')
