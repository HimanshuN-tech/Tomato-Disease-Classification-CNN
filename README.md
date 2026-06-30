# 🍅 Tomato Disease Classification using Convolutional Neural Networks (CNN)

## 📌 Project Overview

This project uses a Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify tomato leaf diseases from images. The model can identify multiple diseases as well as healthy leaves.

The project demonstrates a complete Deep Learning workflow, including data preprocessing, model training, evaluation, visualization, model saving, and prediction on unseen images.

---

## 📂 Dataset

The dataset contains images of tomato leaves belonging to **11 classes**:

* Bacterial Spot
* Early Blight
* Late Blight
* Leaf Mold
* Septoria Leaf Spot
* Spider Mites
* Target Spot
* Tomato Yellow Leaf Curl Virus
* Tomato Mosaic Virus
* Healthy
* Powdery Mildew

Images are resized to **224 × 224** pixels before training.

---

## 🧠 Model Architecture

The CNN consists of:

* Input Layer (224 × 224 × 3)
* Data Augmentation

  * Random Flip
  * Random Rotation
  * Random Zoom
* Conv2D (32 filters)
* MaxPooling2D
* Conv2D (64 filters)
* MaxPooling2D
* Conv2D (128 filters)
* MaxPooling2D
* Flatten Layer
* Dense (128 neurons, ReLU)
* Dropout (0.5)
* Output Layer (11 classes, Softmax)

---

## ⚙️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib

---

## 📈 Training

* Loss Function: Sparse Categorical Crossentropy
* Optimizer: Adam
* Metric: Accuracy
* Epochs: 3

---

## 📊 Features

* Image preprocessing
* Dataset normalization
* Data augmentation
* CNN-based image classification
* Training and validation visualization
* Model saving/loading
* Prediction on new images

---

## 📁 Project Structure

Tomato_Disease_Classification/

├── database/

├── models/

│ └── tomato_disease_model.keras

├── test_images/

│ └── test.jpg

├── src/

│ ├── data_pipeline.py

│ ├── model.py

│ ├── train.py

│ ├── predict.py

│ └── plot_history.py

└── README.md

---

## 🚀 How to Run

### Train the model

python src/train.py

### Predict a new image

python src/predict.py

---

## 📚 What I Learned

* Building CNNs from scratch
* Image preprocessing
* Data normalization
* Data augmentation
* Dropout regularization
* Model evaluation
* Saving and loading TensorFlow models
* Image prediction pipeline

---

## 👨‍💻 Author

**Himanshu Navsare**

This project was built as part of my Deep Learning learning journey.

## Trained Model

The trained `.keras` model is not included in this repository because it exceeds GitHub's 100 MB file size limit.

To generate the trained model:

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Train the model:

```bash
python src/train.py
```

The trained model will be saved automatically inside the `models/` folder.
