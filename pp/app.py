import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# ---------------------------
# 🔹 Load the trained model
# ---------------------------
MODEL_PATH = "plant_disease_model.h5"

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ---------------------------
# 🔹 Load class names automatically
# ---------------------------
DATASET_PATH = "dataset/train"

if os.path.exists(DATASET_PATH):
    CLASS_NAMES = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
else:
    st.error("⚠️ Dataset path not found! Please make sure 'dataset/train' exists.")
    CLASS_NAMES = []

st.title("🌿 Plant Disease Detection App")
st.markdown("""
Upload a leaf image to detect its health condition using your trained model.
""")

# ---------------------------
# 🔹 Upload and predict
# ---------------------------
uploaded_file = st.file_uploader("📤 Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    result_index = np.argmax(prediction)
    result = CLASS_NAMES[result_index] if CLASS_NAMES else "Unknown"
    confidence = np.max(prediction) * 100

    # Display results
    st.success(f"🌱 **Prediction:** {result}")
    st.info(f"💪 **Confidence:** {confidence:.2f}%")

    # Optional probability breakdown
    if CLASS_NAMES:
        st.subheader("🔍 Class Probabilities:")
        prob_dict = {CLASS_NAMES[i]: float(prediction[0][i]) for i in range(len(CLASS_NAMES))}
        st.dataframe({"Class": list(prob_dict.keys()), "Probability": list(prob_dict.values())})
else:
    st.warning("Please upload a leaf image to continue.")
