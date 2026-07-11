"""Visualize the steering-class distribution before and after balancing.

Example
-------
    python tools/plot_distribution.py --data-dir data/ANG
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt

# Make the modules in ../src importable when running this script directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import CLASS_NAMES  # noqa: E402
from data import balance_data, load_log, steering_to_class  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    data = load_log(args.data_dir)
    before = [steering_to_class(v) for v in data["steering"]]
    after = [steering_to_class(v) for v in balance_data(data.copy())["steering"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, values, title in zip(axes, (before, after), ("Before balancing", "After balancing")):
        ax.hist(values, bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.8)
        ax.set_title(title)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(CLASS_NAMES)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
