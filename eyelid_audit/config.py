"""Settings used for the published analysis.

Every value here matches the manuscript. Change them only for new experiments,
and report any change.
"""

# Image preprocessing
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Training
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
FOCAL_GAMMA = 2.0

# Labels
LABEL_MAP = {"benign": 0, "malignant": 1}
CLASS_NAMES = ["Benign", "Malignant"]

# Patient-level split search
SPLIT_TEST_SIZE = 0.15      # fraction of patients held out as the internal test set
SPLIT_VAL_SIZE = 0.20       # fraction of development patients used for validation
SPLIT_N_TRIALS = 1000       # candidate random splits searched
SPLIT_MAX_RATIO_DIFF = 0.15 # max malignant-proportion gap between development and test
PUBLISHED_SPLIT_SEED = 696  # seed of the split used in the paper

# Evaluation
THRESHOLD = 0.5             # equivalent to argmax over the two output logits
BOOTSTRAP_RESAMPLES = 5000  # patient-cluster bootstrap resamples
BOOTSTRAP_SEED = 42

# Sex labels accepted in metadata (English and Indonesian)
MALE_LABELS = {"male", "m", "laki-laki", "l"}
FEMALE_LABELS = {"female", "f", "perempuan", "p"}
