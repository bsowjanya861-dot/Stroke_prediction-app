import streamlit as st
import numpy as np
import cv2
from PIL import Image
from xgboost import XGBClassifier

# -------------------------------
# PAGE SETTINGS
# -------------------------------
st.set_page_config(page_title="Stroke Prediction", layout="centered")
st.title("🧠 Brain Stroke Prediction")

# -------------------------------
# LOAD MODEL (CACHED)
# -------------------------------
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("hybrid_stroke_model.json")
    return model

model = load_model()

# -------------------------------
# MRI CHECK FUNCTION
# -------------------------------
def is_mri_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    std = np.std(gray)

    # MRI images usually have lower variation than colorful images
    if std < 50:
        return True
    else:
        return False

# -------------------------------
# FILE UPLOAD
# -------------------------------
file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "png", "jpeg"])

if file is not None:

    # Show image
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Convert to numpy
    img = np.array(img)

    # -------------------------------
    # FEATURE EXTRACTION
    # -------------------------------
    img_resized = cv2.resize(img, (64, 64))

    pixel_features = img_resized.flatten()

    mean = np.mean(img_resized)
    std = np.std(img_resized)
    maxv = np.max(img_resized)
    minv = np.min(img_resized)

    features = np.hstack([pixel_features, mean, std, maxv, minv])
    features = features.reshape(1, -1)

    # -------------------------------
    # PREDICTION BUTTON
    # -------------------------------
    if st.button("Predict"):

        # Step 1: Check MRI or not
        if not is_mri_image(img):
            st.error("❌ This is NOT a Brain MRI image")
        
        else:
            with st.spinner("Analyzing MRI..."):

                proba = model.predict_proba(features)
                confidence = np.max(proba)
                pred = np.argmax(proba)

            # Step 2: Show result
            if pred == 0:
                result = "Hemorrhagic Stroke"
            elif pred == 1:
                result = "Ischemic Stroke"
            else:
                result = "Unknown"

            st.success(f"🧠 Prediction: {result}")
            st.write(f"Confidence: {confidence*100:.2f}%")

            # Progress bar
            st.progress(int(confidence * 100))
