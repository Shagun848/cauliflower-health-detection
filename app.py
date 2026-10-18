
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image

st.set_page_config(
    page_title="Cauliflower AI",
    page_icon="🌱",
    layout="centered"
)

MODEL_PATH = Path("cauliflower_model.keras")
CLASSES_PATH = Path("classes.json")
IMG_SIZE = (160, 160)

st.title("🌱 Cauliflower Health & Damage Detection")
st.write(
    "Upload a cauliflower leaf image to predict its class and explore "
    "which image regions influenced the prediction using SHAP and LIME."
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.error("Model files are missing from the app repository.")
    st.stop()

try:
    model = load_model()
    class_names = load_classes()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()


def prepare_image(pil_image):
    resized = pil_image.convert("RGB").resize(IMG_SIZE)
    return np.asarray(resized, dtype=np.float32)


def predict_batch(images):
    # Images are RGB values in the 0–255 range.
    # The saved model contains its own Rescaling layer.
    images = np.asarray(images, dtype=np.float32)
    return model.predict(images, verbose=0)


uploaded = st.file_uploader(
    "Choose a cauliflower leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded leaf image",
                 use_container_width=True)

        arr = prepare_image(image)
        probs = predict_batch(np.expand_dims(arr, axis=0))[0]
        idx = int(np.argmax(probs))

        st.subheader("Prediction")
        st.success(class_names[idx])
        st.metric("Model confidence", f"{float(probs[idx]) * 100:.2f}%")

        st.write("### Class scores")
        for name, score in zip(class_names, probs):
            st.write(f"**{name}:** {float(score) * 100:.2f}%")
            st.progress(float(np.clip(score, 0, 1)))

        st.caption(
            "Confidence is the model's score, not a guarantee of correctness. "
            "This educational prototype is not a substitute for expert diagnosis."
        )

        st.divider()
        st.subheader("🔍 Explain the prediction")

        if st.button("Generate SHAP explanation"):
            with st.spinner("Generating SHAP explanation..."):
                try:
                    # A small image masker limits the amount of computation.
                    masker = shap.maskers.Image(
                        "blur(16,16)", arr.shape
                    )
                    explainer = shap.Explainer(
                        predict_batch,
                        masker,
                        output_names=class_names
                    )
                    explanation = explainer(
                        np.expand_dims(arr, axis=0),
                        max_evals=100,
                        batch_size=10
                    )

                    values = explanation.values
                    if values.ndim == 5:
                        selected_values = values[0, :, :, :, idx]
                    elif values.ndim == 4:
                        selected_values = values[0]
                    else:
                        selected_values = values

                
shap.image_plot(
    [selected_values],
    np.expand_dims(arr / 255.0, axis=0),
    show=False
)

fig = plt.gcf()
st.pyplot(fig)
plt.close(fig)


                    st.caption(
                        "SHAP highlights image regions associated with "
                        "the selected class. Colors depend on the "
                        "visualization scale."
                    )
                except Exception as exc:
                    st.error(f"SHAP explanation failed: {exc}")

        if st.button("Generate LIME explanation"):
            with st.spinner("Generating LIME explanation..."):
                try:
                    explainer = lime_image.LimeImageExplainer()
                    explanation = explainer.explain_instance(
                        arr.astype(np.double),
                        predict_batch,
                        top_labels=len(class_names),
                        hide_color=0,
                        num_samples=300
                    )

                    lime_image_result, mask = explanation.get_image_and_mask(
                        idx,
                        positive_only=False,
                        num_features=8,
                        hide_rest=False
                    )

                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(
                        mark_boundaries(lime_image_result / 255.0, mask)
                    )
                    ax.set_title(f"LIME explanation: {class_names[idx]}")
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close(fig)

                    st.caption(
                        "LIME highlights selected superpixels that "
                        "influence the model's prediction. This is an "
                        "approximation, not proof of disease."
                    )
                except Exception as exc:
                    st.error(f"LIME explanation failed: {exc}")

    except Exception as exc:
        st.error(f"Could not process this image: {exc}")

st.divider()
st.caption(
    "Model: MobileNetV2 transfer learning · "
    "Classes: Black Rot, Healthy, Insect Hole"
)
