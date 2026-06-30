from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import random
DATASET_PATH = Path("database")
TRAIN_PATH = DATASET_PATH / "train"
classes = [folder for folder in TRAIN_PATH.iterdir() if folder.is_dir()]
plt.figure(figsize=(15, 12))

for i, disease_folder in enumerate(classes):

    image_files = list(disease_folder.glob("*"))

    random_image = random.choice(image_files)

    image = Image.open(random_image)

    plt.subplot(3, 4, i + 1)
    plt.imshow(image)
    plt.title(disease_folder.name)
    plt.axis("off")

plt.tight_layout()
plt.show()