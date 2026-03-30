import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Swiss Chard Leaf Disease Detection",
    layout="centered"
)

st.title("🌿 Swiss Chard Leaf Disease Classification System")
st.write(
    "This application uses a deep learning model to classify swiss chard leaf images "
    "into healthy or diseased categories and provides treatment recommendations."
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/spinach_leaf_disease_cnn_vs_resnet50.keras"
    )

model = load_model()

# -----------------------------
# Class Labels
# -----------------------------
class_names = [
    "Cercospora Leaf Spot",
    "Healthy",
    "Powdery Mildew"
]

# -----------------------------
# Disease Information
# -----------------------------
disease_info = {
    "Healthy": {
        "description": (
            "The swiss chard leaf shows no visible signs of disease. "
            "Leaf color, texture, and shape appear normal."
        ),
        "treatment": (
            "No treatment required. Continue good agricultural practices, "
            "regular monitoring, and proper irrigation."
        )
    },

    "Powdery Mildew": {
        "description": (
            "Powdery mildew is a fungal disease characterised by white or gray "
            "powder-like growth on the leaf surface. It reduces photosynthesis "
            "and overall plant vigor."
        ),
        "treatment": (
            "• Remove and destroy infected leaves\n"
            "• Improve air circulation between plants\n"
            "• Avoid overhead watering\n"
            "• Apply sulfur-based or potassium bicarbonate fungicides\n"
            "• Use resistant spinach varieties if available"
        )
    },

    "Cercospora Leaf Spot": {
        "description": (
            "Cercospora leaf spot is a fungal disease causing circular to irregular "
            "brown or gray spots with dark margins. Severe infection can lead to "
            "leaf drop and yield loss."
        ),
        "treatment": (
            "• Remove infected plant debris from the field\n"
            "• Practice crop rotation\n"
            "• Avoid excessive leaf wetness\n"
            "• Apply approved fungicides (e.g. copper-based)\n"
            "• Ensure proper field sanitation"
        )
    }
}

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a swiss chard leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width =True)

    # -----------------------------
    # Preprocessing
    # -----------------------------
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # -----------------------------
    # Prediction
    # -----------------------------
    predictions = model.predict(img_array)
    confidence = np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]

    st.subheader("Prediction Result")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2%}")

    # -----------------------------
    # Disease Information Display
    # -----------------------------
    st.subheader("Disease Information")

    st.write(
        f"**Description:**\n\n{disease_info[predicted_class]['description']}"
    )

    st.write(
        f"**Treatment & Management:**\n\n{disease_info[predicted_class]['treatment']}"
    )
