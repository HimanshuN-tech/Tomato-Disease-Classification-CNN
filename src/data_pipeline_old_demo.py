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

# Normalize pixel values from [0, 255] to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
# Data augmentation layer
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# Create the training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)
# Save class names BEFORE applying normalization
class_names = train_dataset.class_names

# Create the validation dataset
valid_dataset = tf.keras.utils.image_dataset_from_directory(
    VALID_PATH,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Apply normalization
train_dataset = train_dataset.map(
    lambda images, labels: (normalization_layer(images), labels)
)

valid_dataset = valid_dataset.map(
    lambda images, labels: (normalization_layer(images), labels)
)
# Display information about the dataset
print("\nClass Names:")
print(class_names)
print("\nNumber of Classes:")
print(len(class_names))
# Verify normalization
for images, labels in train_dataset.take(1):

    print("\nMinimum Pixel Value:", tf.reduce_min(images).numpy())
    print("Maximum Pixel Value:", tf.reduce_max(images).numpy())

    print("\nImage Batch Shape:", images.shape)
    print("Label Batch Shape:", labels.shape)
# Performance optimization
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
valid_dataset = valid_dataset.prefetch(AUTOTUNE)