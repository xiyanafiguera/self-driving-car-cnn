"""Shared constants for the self-driving pipeline."""

# Network input size (NVIDIA PilotNet convention).
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 66

# Vertical crop applied to each raw frame before resizing (keeps the road, drops
# the sky/hood). The raw frames are expected to be at least CROP_BOTTOM px tall.
CROP_TOP = 150
CROP_BOTTOM = 480

# Steering is recorded as a servo value; this one means "go straight".
STEERING_CENTER = 75

# Steering is treated as a 3-class problem rather than continuous regression.
NUM_CLASSES = 3
CLASS_NAMES = ["straight", "left", "right"]  # indices 0, 1, 2

# Data-balancing: cap how many frames may fall in each steering bin so the model
# is not overwhelmed by "straight" driving.
NUM_BINS = 25
SAMPLES_PER_BIN = 400

# Columns of the driving log CSV.
LOG_COLUMNS = ["center", "left", "right", "steering", "throttle", "reverse", "speed"]
