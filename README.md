# Self-Driving Car — CNN Behavioral Cloning

A small robot car that learns to follow a track from camera images, by
**behavioral cloning**: the car is driven manually, (frame, steering) pairs are
recorded, and a convolutional neural network learns to map an image to a driving
action. Steering is framed as a **3-class problem** (straight / left / right).

> Undergraduate capstone project (2021). This is a cleaned implementation
> refactored into Python scripts from the original Colab code. The full write-up
> is in [`report/report.pdf`](report/report.pdf).

## How it works

```
driving log (CSV) ──▶ balance classes ──▶ augment + pre-process ──▶ CNN ──▶ straight / left / right
```

1. **Data** — a driving log references camera frames and their steering value.
   Steering is discretized: `== center → straight`, `< center → left`,
   `> center → right`.
2. **Balancing** — frames are capped per steering bin so the model is not
   dominated by straight-ahead driving.
3. **Augmentation** — random zoom and brightness (pan and flip are available).
4. **Pre-processing** — crop to the road, convert to YUV, Gaussian blur, resize
   to 66×200, scale to `[0, 1]`.
5. **Model** — see below. Trained with sparse categorical cross-entropy (Adam).
6. **Inference** — per frame: pre-process → predict class → driving command.

## Models

**Custom CNN** (NVIDIA-PilotNet-inspired, made lighter for a small dataset):

| Stage | Layers |
| --- | --- |
| Convolutions | 8·(7×7 s2) · 16·(5×5 s2) · 32·(5×5 s2) · 64·(3×3) · 64·(3×3), ReLU |
| Head | Flatten → Dense 100 → 50 → 20 → **3 (softmax)** |

**VGG16 transfer** — a frozen ImageNet VGG16 backbone with a new
`Flatten → Dense 3 (softmax)` head (see [`src/model.py`](src/model.py)).

## Repository structure

```
.
├── src/
│   ├── config.py       # image size, crop, class mapping, balancing params
│   ├── data.py         # load driving log, balance, 3-class labels, split
│   ├── preprocess.py   # crop/YUV/resize + augmentation + batch generator
│   ├── model.py        # custom_cnn() and vgg16_transfer()
│   ├── train.py        # training entry point (CLI)
│   └── infer.py        # run a trained model on a single frame
├── tools/
│   └── plot_distribution.py   # visualize steering-class balance
├── report/report.pdf   # project report
├── requirements.txt
└── README.md
```

## Data

The dataset is **not included**. It is a folder with a driving log and the
frames it references:

```
data/ANG/
├── log_0.csv           # columns: center, left, right, steering, throttle, reverse, speed
└── IMG/                # the camera frames referenced by the log
```

## Usage

```bash
pip install -r requirements.txt

# train the custom CNN (or --model vgg16 for the transfer-learning variant)
python src/train.py --data-dir data/ANG --model custom --epochs 10

# predict the action for a single frame
python src/infer.py --model model.h5 --image frame.jpg

# inspect how balancing changes the class distribution
python tools/plot_distribution.py --data-dir data/ANG
```

## Notes

This is a **reconstruction** of an early (2021) project; the original data and
the on-car deployment code (Raspberry Pi / Arduino motor control) are not part
of this repository. The reported result in the write-up is ~81% validation
accuracy for the custom model. See [`report/report.pdf`](report/report.pdf) for
the full method and discussion.

The write-up's final custom model uses **PReLU** activations; the surviving code
this was refactored from is an earlier **ReLU** version, so the code and the
report differ on that detail.
