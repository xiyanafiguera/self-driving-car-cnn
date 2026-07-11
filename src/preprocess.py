"""Image pre-processing, augmentation and the training batch generator."""

import cv2
import numpy as np
from imgaug import augmenters as iaa

from config import CROP_BOTTOM, CROP_TOP, IMAGE_HEIGHT, IMAGE_WIDTH


# --- augmentation -----------------------------------------------------------

def zoom(image):
    return iaa.Affine(scale=(1, 1.3)).augment_image(image)


def pan(image):
    return iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}).augment_image(image)


def random_brightness(image):
    return iaa.Multiply((0.2, 1.2)).augment_image(image)


def random_flip(image, label):
    """Horizontal flip. Note: if used, the left/right label must be swapped."""
    return cv2.flip(image, 1), label


def random_augment(image_path, label):
    """Read an image and apply a random subset of augmentations."""
    image = cv2.imread(image_path)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = random_brightness(image)
    return image, label


# --- pre-processing ---------------------------------------------------------

def preprocess_image(image):
    """Crop to the road, convert to YUV, blur, resize and scale to [0, 1]."""
    image = image[CROP_TOP:CROP_BOTTOM, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    return image / 255.0


# --- batching ---------------------------------------------------------------

def batch_generator(image_paths, labels, batch_size, is_training):
    """Yield batches of (pre-processed image, class) pairs indefinitely."""
    while True:
        batch_images, batch_labels = [], []
        for _ in range(batch_size):
            index = np.random.randint(len(image_paths))
            if is_training:
                image, label = random_augment(image_paths[index], labels[index])
            else:
                image, label = cv2.imread(image_paths[index]), labels[index]
            batch_images.append(preprocess_image(image))
            batch_labels.append(label)
        yield np.asarray(batch_images), np.asarray(batch_labels)
