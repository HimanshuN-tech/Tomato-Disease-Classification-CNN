from pathlib import Path

# Path to the dataset
DATASET_PATH = Path("database")

# Paths to train and validation folders
TRAIN_PATH = DATASET_PATH / "train"
VALID_PATH = DATASET_PATH / "valid"

# Check if folders exist
print(f"Training folder exists: {TRAIN_PATH.exists()}")
print(f"Validation folder exists: {VALID_PATH.exists()}")
# Find all class folders
classes = [folder.name for folder in TRAIN_PATH.iterdir() if folder.is_dir()]

print("\nDisease Classes:")
for disease in classes:
    print(f"- {disease}")

print(f"\nTotal Classes: {len(classes)}")
print("\nNumber of Images in Each Training Class:\n")

for disease in classes:
    image_count = len(list((TRAIN_PATH / disease).glob("*")))

    print(f"{disease}: {image_count}")