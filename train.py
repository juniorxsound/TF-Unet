import tensorflow as tf
import numpy as np
import os

from src.unet import UNet

# Parameters
batch_size = 32
img_width, img_height, num_channels = 320, 180, 3
lr = 1e-4

# Create input and output placeholders
x = tf.compat.v1.placeholder(tf.float32, shape=[None, img_height, img_width, num_channels], name="x")
y_pred = tf.compat.v1.placeholder(tf.float32, shape=[None, img_height, img_width, num_channels], name="y_pred")

'''
Toy network architecture
'''
# Encoder
conv1 = UNet.conv(x, 3, 3, 64)
conv2 = UNet.conv(conv1, 64, 3, 128)
conv3 = UNet.conv(conv2, 128, 3, 256)

# Decoder
upconv1 = UNet.upconv(conv3, [-1, 45, 80, 128], 256, 128, 128)
upconv2 = UNet.upconv(upconv1, [-1, 90, 160, 64], 128, 64, 64)
upconv2 = UNet.upconv(upconv1, [-1, 180, 320, 3], 64, 3, 3)