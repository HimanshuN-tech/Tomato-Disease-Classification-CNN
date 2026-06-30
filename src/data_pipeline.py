from pathlib import Path
import tensorflow as tf

# Dataset paths
DATASET_PATH = Path("database")
TRAIN_PATH = DATASET_PATH / "train"
VALID_PATH = DATASET_PATH / "valid"

# Image configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Normalize pixel values
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)


def load_datasets():

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_PATH,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    class_names = train_dataset.class_names

    valid_dataset = tf.keras.utils.image_dataset_from_directory(
        VALID_PATH,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    train_dataset = train_dataset.map(
        lambda images, labels: (normalization_layer(images), labels)
    )

    valid_dataset = valid_dataset.map(
        lambda images, labels: (normalization_layer(images), labels)
    )

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(AUTOTUNE)
    valid_dataset = valid_dataset.prefetch(AUTOTUNE)

    return train_dataset, valid_dataset, class_names