# Iris Recognition System Implementation

## Step 1: Import Libraries
# Import necessary libraries for the project.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import cv2
import os

## Step 2: Load and Preprocess Dataset
# Define a function to load images and labels from the dataset directory.
def load_data(dataset_path):
    images = []
    labels = []

    for label in os.listdir(dataset_path):  # Iterate through class directories (1, 2, 3, ...)
        label_path = os.path.join(dataset_path, label)

        if os.path.isdir(label_path):
            for eye_side in ['left', 'right']:  # Process 'left' and 'right' subdirectories
                eye_side_path = os.path.join(label_path, eye_side)

                if os.path.isdir(eye_side_path):
                    for image_file in os.listdir(eye_side_path):
                        image_path = os.path.join(eye_side_path, image_file)

                        # Load and preprocess the image
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        if image is not None:  # Ensure image is successfully loaded
                            image = cv2.resize(image, (128, 128))
                            images.append(image)
                            # Combine the class label and side (e.g., "1_left", "1_right")
                            labels.append(f"{label}_{eye_side}")

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


# Example dataset path (change this to your dataset location)
dataset_path = "A:\FinalYearProject\MMU-Iris-Database"
images, labels = load_data(dataset_path)

# Visualize sample images
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(labels[i])
    plt.axis('off')
plt.tight_layout()
plt.show()


## Step 3: Encode Labels and Split Dataset
# Encode labels to numerical values and split into train and test sets.
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    images, labels_categorical, test_size=0.2, random_state=42
)

# Reshape images to include channel dimension
X_train = X_train.reshape(-1, 128, 128, 1)
X_test = X_test.reshape(-1, 128, 128, 1)


## Step 4: Build the Model
# Define a CNN architecture for iris recognition.
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(labels)), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


## Step 5: Train the Model
# Train the model using the training dataset.
history = model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test), 
    epochs=10, 
    batch_size=32
)


## Step 6: Evaluate the Model
# Evaluate the model and visualize results.
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()


## Step 7: Predictions and Confusion Matrix
# Make predictions and display confusion matrix.
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print(classification_report(y_true_classes, y_pred_classes))

conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


## Step 8: Save the Model
# Save the trained model for deployment.
model.save("iris_recognition_model.h5")
print("Model saved as iris_recognition_model.h5")
