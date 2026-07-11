"""Run a trained model on a single frame and print the predicted action.

On the car, this is the step that runs per camera frame: pre-process the
frame, ask the model for a class, and turn it into a driving command.

Example
-------
    python src/infer.py --model model.h5 --image frame.jpg
"""

import argparse

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from config import CLASS_NAMES
from preprocess import preprocess_image


def predict(model, image_bgr):
    """Return (class_name, confidence) for a single BGR image."""
    x = preprocess_image(image_bgr)
    probabilities = model.predict(np.expand_dims(x, axis=0), verbose=0)[0]
    index = int(np.argmax(probabilities))
    return CLASS_NAMES[index], float(probabilities[index])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to a saved .h5 model")
    parser.add_argument("--image", required=True, help="path to an input frame")
    args = parser.parse_args()

    model = load_model(args.model)
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")

    label, confidence = predict(model, image)
    print(f"{label} ({confidence:.1%})")


if __name__ == "__main__":
    main()
