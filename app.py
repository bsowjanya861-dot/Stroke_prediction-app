import streamlit as st
import numpy as np
import cv2
from PIL import Image
from xgboost import XGBClassifier

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Brain Stroke Detection", layout="centered")

st.title("🧠 Brain Stroke Detection")
st.write("Upload Brain MRI Image")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("hybrid_stroke_model.json")
    return model

model = load_model()

# -------------------- UPLOAD --------------------
file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = np.array(img)

    # -------------------- BASIC VALIDATION --------------------
    if img is None:
        st.error("❌ Invalid image file")
        st.stop()

    # Resize
    img = cv2.resize(img, (64, 64))

    # Normalize
    img = img / 255.0

    # -------------------- IMAGE ANALYSIS --------------------
    mean = np.mean(img)
    std = np.std(img)

    # 🚨 Reject clearly wrong images
    if mean < 0.05 or mean > 0.95:
        st.error("❌ Invalid Image (Lighting issue / not MRI)")
        st.stop()

    if std < 0.05:
        st.error("❌ Invalid Image (Too plain / not MRI)")
        st.stop()

    # -------------------- FEATURE EXTRACTION --------------------
    features = img.flatten()
    features = features.reshape(1, -1)

    # -------------------- PREDICT --------------------
    if st.button("🔍 Predict"):

        proba = model.predict_proba(features)
        confidence = float(np.max(proba))
        pred = int(np.argmax(proba))

        # 🚨 Strong rejection rule
        if confidence < 0.85:
            st.error("❌ Invalid Image (Not a Brain MRI)")
        
        else:
            if pred == 0:
                st.error("⚠️ Hemorrhagic Stroke Detected")
            else:
                st.success("✅ Ischemic Stroke Detected")

            st.write(f"Confidence: {confidence:.2f}")
