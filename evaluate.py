"""TF-UNet written by @juniorxsound <https://orfleisher.com>"""

# Dependencies
from tensorflow import keras  # pylint: disable=import-error
import numpy as np  # pylint: disable=import-error

# Components
from unet.unet import UNet
from unet.dataset import ShapesDataset

# Some constants
train_num_samples = 100
eval_num_samples = 10
image_width, image_height, num_channels = 128, 128, 3

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(eval_num_samples, image_height, image_width)
dataset_val.prepare()

# Organize the evaluation set in two numpy arrays
image = [dataset_val.load_image(image_id).astype(
    np.float16) for image_id in dataset_val.image_ids]
masks = []
for image_id in dataset_val.image_ids:
    mask, classes = dataset_val.load_mask(image_id)

    # Treshold the segmentation images to get b&w alpha mask
    mask_treshold = np.expand_dims(
        np.where(mask[:, :, 0] > 0, 255, 0).astype(np.float16),
        axis=3)
    masks.append(mask_treshold)

# Create U-Network and load weights
unet = UNet()
unet.compile(optimizer="adam",
             loss="binary_crossentropy",
             metrics=["accuracy"])
unet.load_weights("./weights/tf_unet_toy_network")
loss, acc = unet.evaluate(np.array(image),  np.array(masks), verbose=2)
print("Restored model and ran evaluation, accuracy: {:5.2f}%".format(100*acc))
