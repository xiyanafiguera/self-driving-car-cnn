"""Model definitions.

Two variants share the same 3-class steering head:

* ``custom_cnn``   -- an NVIDIA-PilotNet-inspired network trained from scratch,
  made lighter (fewer early filters, a wider first kernel) to suit a small
  self-collected dataset.
* ``vgg16_transfer`` -- a transfer-learning baseline with a frozen ImageNet
  VGG16 backbone and a new classification head.

Both are trained with sparse categorical cross-entropy (the categorical
negative log-likelihood for integer labels).
"""

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from config import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES

INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)


def custom_cnn(learning_rate: float = 1e-4) -> Sequential:
    model = Sequential([
        Conv2D(8, (7, 7), strides=(2, 2), activation="relu", padding="valid", input_shape=INPUT_SHAPE),
        Conv2D(16, (5, 5), strides=(2, 2), activation="relu", padding="valid"),
        Conv2D(32, (5, 5), strides=(2, 2), activation="relu", padding="valid"),
        Conv2D(64, (3, 3), activation="relu", padding="valid"),
        Conv2D(64, (3, 3), activation="relu", padding="valid"),
        Flatten(),
        Dense(100, activation="relu"),
        Dense(50, activation="relu"),
        Dense(20, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model


def vgg16_transfer(learning_rate: float = 1e-3) -> Model:
    input_layer = Input(shape=INPUT_SHAPE)
    base = VGG16(weights="imagenet", include_top=False, input_tensor=input_layer)
    for layer in base.layers:
        layer.trainable = False

    x = Flatten()(base.output)
    x = Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = Model(base.input, x)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model


def build(name: str = "custom"):
    """Factory used by ``train.py``."""
    return custom_cnn() if name == "custom" else vgg16_transfer()
