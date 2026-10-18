import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Cauliflower AI", page_icon="🌱", layout="centered")

MODEL_PATH = Path("cauliflower_model.keras")
CLASSES_PATH = Path("classes.json")
IMG_SIZE = (160, 160)

st.title("🌱 Cauliflower Health & Damage Detection")
st.write(
    "Upload a cauliflower leaf image to see the model's predicted class. "
    "This is an educational prototype, not a substitute for expert diagnosis."
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.error("Model files are missing. Add cauliflower_model.keras and classes.json to the app repository.")
    st.stop()

try:
    model = load_model()
    class_names = load_classes()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded leaf image", use_container_width=True)

        resized = image.resize(IMG_SIZE)
        arr = np.asarray(resized, dtype=np.float32)
        batch = np.expand_dims(arr, axis=0)

        # The saved model includes a Rescaling layer, so pass RGB pixels in the 0–255 range.
        probs = model.predict(batch, verbose=0)[0]
        idx = int(np.argmax(probs))

        st.subheader("Prediction")
        st.success(f"{class_names[idx]}")
        st.metric("Model confidence", f"{float(probs[idx]) * 100:.2f}%")

        st.write("Class scores")
        for name, score in zip(class_names, probs):
            st.write(f"**{name}:** {float(score) * 100:.2f}%")
            st.progress(float(np.clip(score, 0, 1)))

        st.caption(
            "Confidence is the model's score, not a guarantee of correctness. "
            "Image quality, lighting, and differences from the training dataset can affect results."
        )
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")

st.divider()
st.caption("Model: MobileNetV2 transfer learning · Classes: Black Rot, Healthy, Insect Hole")
