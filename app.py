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
# LOAD MODEL (ONLY ONCE)
# -------------------------------
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("hybrid_stroke_model.json")
    return model

model = load_model()

# -------------------------------
# STRONG MRI CHECK FUNCTION
# -------------------------------
def is_mri_image(img):
    img = np.array(img)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Channel similarity check (MRI = grayscale)
    b, g, r = cv2.split(img)
    diff_rg = np.mean(np.abs(r - g))
    diff_rb = np.mean(np.abs(r - b))
    diff_gb = np.mean(np.abs(g - b))

    # Texture variation
    std = np.std(gray)

    # FINAL CONDITION (combined check)
    if (diff_rg < 10 and diff_rb < 10 and diff_gb < 10) and (std < 80):
        return True
    else:
        return False

# -------------------------------
# FILE UPLOAD
# -------------------------------
file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "png", "jpeg"])

if file is not None:

    # Show uploaded image
    img_pil = Image.open(file).convert("RGB")
    st.image(img_pil, caption="Uploaded Image", use_container_width=True)

    # Convert to numpy
    img = np.array(img_pil)

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

        # 🚨 STEP 1: MRI VALIDATION
        if not is_mri_image(img):
            st.error("❌ This is NOT a Brain MRI image. Please upload correct scan.")
        
        else:
            # 🚀 STEP 2: MODEL PREDICTION
            with st.spinner("Analyzing MRI..."):
                proba = model.predict_proba(features)
                confidence = np.max(proba)
                pred = np.argmax(proba)

            # 🚀 STEP 3: RESULT
            if pred == 0:
                result = "Hemorrhagic Stroke"
            elif pred == 1:
                result = "Ischemic Stroke"
            else:
                result = "Unknown"

            # 🚀 STEP 4: DISPLAY
            st.success(f"🧠 Prediction: {result}")
            st.write(f"Confidence: {confidence*100:.2f}%")

            # Progress bar
            st.progress(int(confidence * 100))
