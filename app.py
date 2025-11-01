import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

# Load trained model
model = load_model("mnist_cnn_model.keras")

st.title("🧠 MNIST Digit Recognizer")
st.write("Upload a black & white image of a handwritten digit (0–9) to predict it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale

    # Resize and center the image
    image = ImageOps.fit(image, (28, 28), Image.Resampling.LANCZOS)
    
    # Convert image to numpy array
    img_array = np.array(image)

    # Invert if background is white
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    # Normalize the image
    img_array = img_array / 255.0

    # Reshape to model input
    img_array = img_array.reshape(1, 28, 28, 1)

    # Show processed image
    st.image(image, caption='Processed Image (28x28)', width=150)

    # Predict
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    st.success(f"Predicted Digit: {predicted_label}")
