"""Train a steering model by behavioral cloning.

Examples
--------
    python src/train.py --data-dir data/ANG --model custom --epochs 10
    python src/train.py --data-dir data/ANG --model vgg16 --epochs 1
"""

import argparse

from data import balance_data, load_img_steering, load_log, split
from model import build
from preprocess import batch_generator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="folder with log_0.csv and an IMG/ subfolder")
    parser.add_argument("--model", choices=["custom", "vgg16"], default="custom")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=300)
    parser.add_argument("--val-steps", type=int, default=200)
    parser.add_argument("--out", default="model.h5")
    args = parser.parse_args()

    data = balance_data(load_log(args.data_dir))
    image_paths, labels = load_img_steering(f"{args.data_dir}/IMG", data)
    x_train, x_valid, y_train, y_valid = split(image_paths, labels)
    print(f"Training samples: {len(x_train)}  |  Validation samples: {len(x_valid)}")

    model = build(args.model)
    model.summary()

    model.fit(
        batch_generator(x_train, y_train, args.batch_size, is_training=True),
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        validation_data=batch_generator(x_valid, y_valid, args.batch_size, is_training=False),
        validation_steps=args.val_steps,
        verbose=1,
    )

    model.save(args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
