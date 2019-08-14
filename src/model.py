import tensorflow as tf

__DEBUG__ = True


class DepthNet(object):

    '''
    Utility section for stuff that get's reused when building the network
    '''
    @staticmethod
    def create_weights(shape, name='W'):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05), name=name)

    @staticmethod
    def create_biases(size, name='B'):
        return tf.Variable(tf.constant(0.05, shape=[size]), name=name)

    '''
    Network architecture blocks
    '''
    def conv_layer(input,
                   input_channels,
                   output_channels,
                   kernel_size,
                   layer_stride=[1, 1, 1, 1],
                   use_relu=True,
                   name='conv'):
        with tf.name_scope(name):
            # We shall define the weights that will be trained using create_weights function.
            weights = DepthNet.create_weights(
                shape=[output_channels, output_channels, input_channels, kernel_size])
            print(weights.get_shape())
            # We create biases using the create_biases function. These are also trained.
            biases = DepthNet.create_biases(kernel_size)

            # Creating the convolutional layer
            layer = tf.nn.conv2d(input=input,
                                 filter=weights,
                                 strides=layer_stride,
                                 padding='SAME')

            # Add the biases to the layer
            layer += biases

            # Output layer is fed to Relu which is the activation function for us.
            if use_relu is True:
                layer = tf.nn.relu(layer)

            # Create summaries for TensorBoard
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("biases", biases)
            tf.summary.histogram("activitions", layer)

            # Debug prints
            if (__DEBUG__):
                print('Layer shape: {}'.format(layer.get_shape()))

            return layer

    def upconv_layer(input,
                     output_shape,
                     input_channels,
                     output_channels,
                     kernel_size,
                     layer_stride=[1, 2, 2, 1],
                     use_relu=True,
                     name='upconv'):
        with tf.name_scope(name):
            # We shall define the weights that will be trained using create_weights function.
            weights = DepthNet.create_weights(
                shape=[output_channels, output_channels, input_channels, kernel_size])

            # We create biases using the create_biases function. These are also trained.
            biases = DepthNet.create_biases(kernel_size)

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
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("biases", biases)
            tf.summary.histogram("activitions", layer)

            # Debug prints
            if (__DEBUG__):
                print('Layer shape: {}'.format(layer.get_shape()))

            return layer

    @staticmethod
    def create_flatten_layer(layer, name='flatten'):
        with tf.name_scope(name):
            # We know that the shape of the layer will be [batch_size img_size img_size num_channels]
            # But let's get it from the previous layer.
            layer_shape = layer.get_shape()

            # Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
            num_features = layer_shape[1:4].num_elements()

            # Now, we Flatten the layer so we shall have to reshape to num_features
            layer = tf.reshape(layer, [-1, num_features])

            print(layer.get_shape())

            return layer

    @staticmethod
    def create_fc_layer(input,
                        num_inputs,
                        num_outputs,
                        use_relu=True,
                        name='fc'):
        with tf.name_scope(name):
            # Let's define trainable weights and biases.
            weights = NetworkFactory.create_weights(
                shape=[num_inputs, num_outputs])
            biases = NetworkFactory.create_biases(num_outputs)

            # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
            layer = tf.matmul(input, weights) + biases
            if use_relu:
                layer = tf.nn.relu(layer)

            print(layer.get_shape())

            return layer


'''
Used for unit testing the creation of each layer ⚠️ should not be used by directly, see train.py for the proper interface
'''
if __name__ == "__main__":

    if __DEBUG__ is False:
        __DEBUG__ = True

    # Create a test input layer
    x = tf.placeholder(tf.float32, shape=[None, 180, 320, 3], name="x")

    # Create down conv layer
    print('\nCreating down conv layer (batch_dim, height, width, depth)')
    print('Input shape: {}'.format(x.get_shape()))
    layer = DepthNet.conv_layer(x, 3, 3, 32, layer_stride=[1, 2, 2, 1])

    # Assert shapes
    assert int(x.get_shape()[1]) / 2 == layer.get_shape()[1]
    assert int(x.get_shape()[2]) / 2 == layer.get_shape()[2]
    print('✅ Down conv reduces height and width by a factor of 2')

    # Create regular conv layer
    print('\nCreating regular conv layer (batch_dim, height, width, depth)')
    print('Input shape: {}'.format(x.get_shape()))
    layer = DepthNet.conv_layer(x, 3, 3, 3)

    # Assert shapes
    assert int(x.get_shape()[1]) == layer.get_shape()[1]
    assert int(x.get_shape()[2]) == layer.get_shape()[2]
    print('✅ Regular conv keeps height, width and depth the SAME')

    # Create upconv (transposed conv) layer
    print('\nCreating upconv layer (batch_dim, height, width, depth)')
    print('Input shape: {}'.format(x.get_shape()))
    layer = DepthNet.upconv_layer(x, [-1, 360, 640, 3], 3, 3, 3)

    # Assert shapes
    # assert int(x.get_shape()[1]) == layer.get_shape()[1]
    # assert int(x.get_shape()[2]) == layer.get_shape()[2]
    # print('✅ Regular conv keeps height, width and depth the SAME')
