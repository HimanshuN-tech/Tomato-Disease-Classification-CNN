import tensorflow as tf
from pathlib import Path
import numpy as np

# Path to the saved model
MODEL_PATH = Path("models/tomato_disease_model.keras")

# Path to the test image
IMAGE_PATH = Path("test_images/test.jpg")
from data_pipeline import load_datasets

# Load class names
_, _, class_names = load_datasets()

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")
# Load the image
image = tf.keras.utils.load_img(
    IMAGE_PATH,
    target_size=(224, 224)
)

print("Image loaded successfully!")
# Convert image to array
image_array = tf.keras.utils.img_to_array(image)

print("Image Shape:", image_array.shape)
# Normalize pixel values from [0, 255] to [0, 1]
image_array = image_array / 255.0

print("Minimum Pixel Value:", image_array.min())
print("Maximum Pixel Value:", image_array.max())
# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

print("Image Shape After Batch Dimension:", image_array.shape)
# Predict the class probabilities
predictions = model.predict(image_array)

print("Prediction Shape:", predictions.shape)
print("Predictions:", predictions)
# Get the predicted class index
predicted_index = np.argmax(predictions)

# Get the predicted class name
predicted_class = class_names[predicted_index]

# Get confidence score
confidence = predictions[0][predicted_index] * 100

print("\nPrediction Result")
print("-----------------")
print("Predicted Disease:", predicted_class)
print(f"Confidence: {confidence:.2f}%")