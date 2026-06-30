from model import create_model
from data_pipeline import load_datasets
from plot_history import plot_history

# Load datasets
train_dataset, valid_dataset, class_names = load_datasets()

# Create model
model = create_model()

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display model architecture
model.summary()
# Train the model
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=3
)
print(history.history.keys())
# Plot Accuracy and Loss graphs
print("Step 1: Before plotting")
plot_history(history)

print("Step 2: After plotting")

print("Step 3: Before saving")
model.save("models/tomato_disease_model.keras")

print("Step 4: Model saved successfully!")