import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps


model = load_model("mnist_cnn_model.keras")

st.title("🧠 MNIST Digit Recognizer")
st.write("Upload a black & white image of a handwritten digit (0–9) to predict it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert('L') 
    image = ImageOps.invert(image)  
    image = image.resize((28, 28))  
    img_array = np.array(image)
    img_array = img_array.reshape(1, 28, 28, 1)
    img_array = img_array / 255.0

    st.image(image, caption='Uploaded Image', width=150)

    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    st.success(f"Predicted Digit: {predicted_label}")
