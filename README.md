 🧠 MNIST Digit Recognizer (Streamlit + TensorFlow)

An interactive **web app** built with **Streamlit** and a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**.  
Users can upload a handwritten digit image (0–9) and the app predicts the digit in real time.

---

## 📁 Project Structure

MNIST_classification/
│
├── mnist_cnn_model.keras # Pre-trained CNN model
├── app.py # Streamlit web app script
├── MNIST_.py # Training script (optional)
├── README.md # Project documentation
└── requirements.txt (optional)

yaml
Copy code

---

## 🚀 Features

- Upload handwritten digit images (JPG, JPEG, PNG)  
- Real-time prediction using trained CNN  
- Streamlit-based web interface  
- Uses TensorFlow/Keras for deep learning  

---

## 🧩 Model Overview

The model is trained on the **MNIST dataset** (60,000 training + 10,000 test images).

| Layer (type) | Output Shape | Parameters |
|---------------|---------------|-------------|
| Conv2D (32 filters, 3×3) | (26, 26, 32) | 320 |
| MaxPooling2D (2×2) | (13, 13, 32) | 0 |
| Conv2D (64 filters, 3×3) | (11, 11, 64) | 18496 |
| MaxPooling2D (2×2) | (5, 5, 64) | 0 |
| Conv2D (64 filters, 3×3) | (3, 3, 64) | 36928 |
| Flatten | (576) | 0 |
| Dense (64 units, ReLU) | (64) | 36928 |
| Dropout (0.5) | (64) | 0 |
| Dense (10 units, Softmax) | (10) | 650 |

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/manoj07-ai/MNIST_Classification
Install dependencies:

bash
Copy code
pip install tensorflow streamlit pillow numpy
🧠 Run the Web App
bash
Copy code
streamlit run app.py
Then open the local URL shown in the terminal (e.g. http://localhost:8501/).

💡 How It Works
Upload an image of a handwritten digit (0–9).

The app preprocesses the image:

Converts to grayscale

Inverts colors (white digit on black background)

Resizes to 28×28 pixels

Normalizes pixel values

The trained CNN model predicts the digit.

The predicted label is displayed instantly.

🧰 Technologies Used
Python

TensorFlow / Keras

Streamlit

Pillow (PIL)

NumPy

📈 Model Performance
Training Accuracy: ~98.2%

Validation Accuracy: ~99.0%

Loss Function: Sparse Categorical Crossentropy

Optimizer: Adam

👨‍💻 Author
Manoj Kumar Marupalli
🎓 AI & Deep Learning Enthusiast
📧 Open to collaboration and feedback!

🏷️ License
This project is licensed under the MIT License — feel free to use and modify it.

