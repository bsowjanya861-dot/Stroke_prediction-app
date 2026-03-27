import streamlit as st
import numpy as np
import cv2
from PIL import Image
from xgboost import XGBClassifier

# -------------------- PAGE SETTINGS --------------------
st.set_page_config(
    page_title="Brain Stroke Detection",
    page_icon="🧠",
    layout="centered"
)

# -------------------- TITLE --------------------
st.title("🧠 Brain Stroke Detection App")
st.markdown("Upload a brain MRI image to predict stroke type")

# -------------------- SIDEBAR --------------------
st.sidebar.title("About")
st.sidebar.info(
    "This app uses XGBoost model to classify MRI images into:\n"
    "- Hemorrhagic Stroke\n"
    "- Ischemic Stroke\n\n"
    "⚠️ Not for medical use"
)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("hybrid_stroke_model.json")
    return model

model = load_model()

# -------------------- FILE UPLOAD --------------------
file = st.file_uploader("📤 Upload Brain MRI Image", type=["jpg", "png", "jpeg"])

# -------------------- PROCESS --------------------
if file is not None:

    try:
        # Read image
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Convert to numpy
        img = np.array(img)

        # Resize (IMPORTANT: must match training)
        img = cv2.resize(img, (64, 64))

        # Feature extraction
        pixel_features = img.flatten()

        mean = np.mean(img)
        std = np.std(img)
        maxv = np.max(img)
        minv = np.min(img)

        features = np.hstack([pixel_features, mean, std, maxv, minv])
        features = features.reshape(1, -1)

        # -------------------- PREDICT --------------------
        if st.button("🔍 Predict"):

            proba = model.predict_proba(features)
            confidence = np.max(proba)
            pred = np.argmax(proba)

            # -------------------- CONFIDENCE CHECK --------------------
            if confidence < 0.7:
                st.warning("⚠️ This does not look like a valid brain MRI image")
                st.write(f"Confidence: {confidence:.2f}")

            else:
                if pred == 0:
                    st.error("⚠️ Hemorrhagic Stroke Detected")
                else:
                    st.success("✅ Ischemic Stroke Detected")

                st.write(f"Confidence: {confidence:.2f}")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
