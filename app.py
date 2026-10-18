

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


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Cauliflower AI",
    page_icon="🌱",
    layout="centered"
)

IMG_SIZE = (160, 160)
MODEL_PATH = Path("cauliflower_model.keras")
CLASSES_PATH = Path("classes.json")

st.title("🌱 Cauliflower Health & Damage Detection")

st.write(
    "Upload a cauliflower leaf image to predict its class "
    "and explore which image regions influenced the prediction "
    "using SHAP and LIME."
)


# -----------------------------
# Load model and class names
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_classes():
    with open(CLASSES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.error(
        "Model files are missing. Check that "
        "cauliflower_model.keras and classes.json are in your repository."
    )
    st.stop()

try:
    model = load_model()
    class_names = load_classes()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()


# -----------------------------
# Image preparation and prediction
# -----------------------------
def prepare_image(pil_image):
    resized = pil_image.convert("RGB").resize(IMG_SIZE)
    return np.asarray(resized, dtype=np.float32)


def predict_batch(images):
    """
    Input images use RGB values from 0 to 255.
    The saved model contains its own Rescaling layer.
    """
    images = np.asarray(images, dtype=np.float32)
    return model.predict(images, verbose=0)


# -----------------------------
# Upload image
# -----------------------------
uploaded = st.file_uploader(
    "Choose a cauliflower leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")

        st.image(
            image,
            caption="Uploaded leaf image",
            use_container_width=True
        )

        arr = prepare_image(image)
        batch_image = np.expand_dims(arr, axis=0)

        probs = predict_batch(batch_image)[0]
        idx = int(np.argmax(probs))

        # Prediction
        st.subheader("Prediction")
        st.success(f"Predicted class: {class_names[idx]}")

        st.metric(
            "Model confidence",
            f"{float(probs[idx]) * 100:.2f}%"
        )

        # Class scores
        st.write("### Class scores")

        for name, score in zip(class_names, probs):
            st.write(f"**{name}:** {float(score) * 100:.2f}%")
            st.progress(float(np.clip(score, 0, 1)))

        st.caption(
            "Confidence is the model's score, not a guarantee "
            "of correctness. This educational prototype is not "
            "a substitute for expert diagnosis."
        )

        # -----------------------------
        # SHAP and LIME explanations
        # -----------------------------
        st.divider()
        st.subheader("🔍 Explain the prediction")

        # SHAP
        if st.button("Generate SHAP explanation"):
            with st.spinner("Generating SHAP explanation..."):
                try:
                    masker = shap.maskers.Image(
                        "blur(16,16)",
                        arr.shape
                    )

                    explainer = shap.Explainer(
                        predict_batch,
                        masker,
                        output_names=class_names
                    )

                    explanation = explainer(
                        batch_image,
                        max_evals=100,
                        batch_size=10
                    )

                    values = np.asarray(explanation.values)

                    # Select SHAP values for the predicted class.
                    if values.ndim == 5:
                        selected_values = values[0, :, :, :, idx]
                    elif values.ndim == 4:
                        selected_values = values[0]
                    else:
                        raise ValueError(
                            f"Unexpected SHAP output shape: {values.shape}"
                        )

                    # Ensure selected values have H x W x C shape.
                    if selected_values.shape != arr.shape:
                        raise ValueError(
                            "Unexpected SHAP image shape: "
                            f"{selected_values.shape}"
                        )

                    # Convert SHAP values into a simple heatmap.
                    heatmap = np.mean(
                        np.abs(selected_values),
                        axis=-1
                    )

                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(arr.astype(np.uint8))
                    ax.imshow(
                        heatmap,
                        cmap="jet",
                        alpha=0.45
                    )
                    ax.set_title(
                        f"SHAP explanation: {class_names[idx]}"
                    )
                    ax.axis("off")

                    st.pyplot(fig)
                    plt.close(fig)

                    st.caption(
                        "The heatmap shows regions with larger absolute "
                        "SHAP values for the selected class. It is an "
                        "approximate explanation, not proof of disease."
                    )

                except Exception as exc:
                    st.error(f"SHAP explanation failed: {exc}")

        # LIME
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

                    lime_image_result, mask = (
                        explanation.get_image_and_mask(
                            idx,
                            positive_only=False,
                            num_features=8,
                            hide_rest=False
                        )
                    )

                    fig, ax = plt.subplots(figsize=(8, 6))

                    ax.imshow(
                        mark_boundaries(
                            lime_image_result / 255.0,
                            mask
                        )
                    )

                    ax.set_title(
                        f"LIME explanation: {class_names[idx]}"
                    )
                    ax.axis("off")

                    st.pyplot(fig)
                    plt.close(fig)

                    st.caption(
                        "LIME highlights selected image regions that "
                        "influence the model's prediction. This is an "
                        "approximation, not proof of disease."
                    )

                except Exception as exc:
                    st.error(f"LIME explanation failed: {exc}")

    except Exception as exc:
        st.error(f"Could not process this image: {exc}")


# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "Model: MobileNetV2 transfer learning · "
    "Classes: Black Rot, Healthy, Insect Hole"
)
