import tensorflow as tf

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 11
# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
], name="data_augmentation")

def create_model():
    model = tf.keras.Sequential([

        # Input layer
        tf.keras.layers.Input(shape=(224, 224, 3)),
        # Data Augmentation
        data_augmentation,

            # Block 1
tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation="relu",
    padding="same"
),
tf.keras.layers.MaxPooling2D(),

# Block 2
tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    activation="relu",
    padding="same"
),
tf.keras.layers.MaxPooling2D(),

# Block 3
tf.keras.layers.Conv2D(
    filters=128,
    kernel_size=(3, 3),
    activation="relu",
    padding="same"
),
tf.keras.layers.MaxPooling2D(),

        # Flatten
        tf.keras.layers.Flatten(),

        # Fully Connected Layer
        tf.keras.layers.Dense(128, activation='relu'),

        # Dropout Layer
        tf.keras.layers.Dropout(0.5),

        # Output Layer
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    return model