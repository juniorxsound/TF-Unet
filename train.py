"""TF-UNet written by @juniorxsound <https://orfleisher.com>"""

# Dependencies
import numpy as np  # pylint: disable=import-error
from datetime import datetime
from tensorflow import keras  # pylint: disable=import-error

# Components
from unet.dataset import ShapesDataset
from unet.unet import UNet

# Some constants
train_num_samples = 100
image_width, image_height = 128, 128
num_ecpochs = 1
batch_size = 1

# Tensorboard callback and logging directory
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, update_freq='batch')

# Training dataset
dataset_train = ShapesDataset()
dataset_train.load_shapes(train_num_samples, image_height, image_width)
dataset_train.prepare()

# Create U-Network
unet = UNet()
unet.compile(optimizer="adam",
             loss="binary_crossentropy",
             metrics=["accuracy"])

# Organize the dataset in two numpy arrays
image = [dataset_train.load_image(image_id).astype(
    np.float16) for image_id in dataset_train.image_ids]
masks = []
for image_id in dataset_train.image_ids:
    mask, classes = dataset_train.load_mask(image_id)

    # Treshold the segmentation images to get b&w alpha mask
    mask_treshold = np.expand_dims(
        np.where(mask[:, :, 0] > 0, 255, 0).astype(np.float16),
        axis=3)
    masks.append(mask_treshold)

# Train
history = unet.fit(np.array(image),  # X
                   np.array(masks),  # Y
                   batch_size=batch_size,
                   epochs=num_ecpochs,
                   callbacks=[tensorboard_callback],
                   verbose=1)

# Save the weights into a tf.SavedModel
unet.save_weights("./weights/tf_unet_toy_network")
