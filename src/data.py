"""Driving-log loading, class balancing and the image/label split.

The driving log is a CSV (``log_0.csv``) recorded while driving the car
manually; each row references a camera frame and its steering value. Steering
is turned into three classes (straight / left / right).
"""

import ntpath
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from config import (
    LOG_COLUMNS,
    NUM_BINS,
    SAMPLES_PER_BIN,
    STEERING_CENTER,
)


def path_leaf(path: str) -> str:
    """Return just the file name from a (possibly absolute) path."""
    return ntpath.split(path)[1]


def load_log(data_dir: str, log_file: str = "log_0.csv") -> pd.DataFrame:
    """Load the driving log and reduce the ``center`` column to file names."""
    data = pd.read_csv(os.path.join(data_dir, log_file), names=LOG_COLUMNS)
    data["center"] = data["center"].apply(path_leaf)
    return data


def balance_data(data: pd.DataFrame, num_bins: int = NUM_BINS,
                 samples_per_bin: int = SAMPLES_PER_BIN) -> pd.DataFrame:
    """Cap the number of samples per steering bin to reduce the straight bias."""
    _, bins = np.histogram(data["steering"], num_bins)
    to_remove = []
    for j in range(num_bins):
        in_bin = [
            i for i in range(len(data))
            if bins[j] <= data["steering"].iloc[i] <= bins[j + 1]
        ]
        in_bin = shuffle(in_bin)
        to_remove.extend(in_bin[samples_per_bin:])
    data = data.drop(data.index[to_remove]).reset_index(drop=True)
    return data


def steering_to_class(value: float, center: int = STEERING_CENTER) -> int:
    """Map a steering value to a class: 0 = straight, 1 = left, 2 = right."""
    if value == center:
        return 0
    return 1 if value < center else 2


def load_img_steering(img_dir: str, data: pd.DataFrame):
    """Build parallel arrays of image paths and integer steering classes."""
    image_paths, labels = [], []
    for i in range(len(data)):
        row = data.iloc[i]
        image_paths.append(os.path.join(img_dir, row["center"].strip()))
        labels.append(steering_to_class(float(row["steering"])))
    return np.asarray(image_paths), np.asarray(labels)


def split(image_paths, labels, test_size: float = 0.2, seed: int = 6):
    """Train/validation split."""
    return train_test_split(image_paths, labels, test_size=test_size, random_state=seed)
