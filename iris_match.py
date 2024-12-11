import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

# Function to load and preprocess an image
def load_and_preprocess_image(img_path, input_width, input_height):
    # Load the image in grayscale as your model is trained on grayscale images
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Ensure the image is not None
    if img is not None:
        img = cv2.resize(img, (input_width, input_height))  # Resize to model input shape
        img = img.astype("float32") / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=-1)  # Add channel dimension (height, width, channels)
        img = np.expand_dims(img, axis=0)  # Add batch dimension (batch_size, height, width, channels)
    return img

# Load the trained model (replace with your actual model file path)
model = load_model("iris_recognition_model.h5")

# Define input dimensions based on your model
input_width = 128  # The width of the input image used during training
input_height = 128  # The height of the input image used during training

# Paths to the left and right images (update these paths with the actual paths)
left_image_path = r"A:\FinalYearProject\sample-iris\aeval5.bmp"  # Replace with the actual path
right_image_path = r"A:\FinalYearProject\sample-iris\aevar5.bmp"  # Replace with the actual path

# Load and preprocess the images
left_image = load_and_preprocess_image(left_image_path, input_width, input_height)
right_image = load_and_preprocess_image(right_image_path, input_width, input_height)

# Get predictions for both images
left_pred = model.predict(left_image)
right_pred = model.predict(right_image)

# Compute similarity using cosine similarity between the predictions
similarity = cosine_similarity(left_pred, right_pred)

# Set a similarity threshold (adjust based on your data and model)
threshold = 0.9  # You can adjust this value depending on how sensitive you want the matching to be

# Print the similarity score and check if the images match
print(f"Similarity score: {similarity[0][0]:.4f}")
if similarity[0][0] > threshold:
    print("The images match!")
else:
    print("The images do not match.")
