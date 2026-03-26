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
# LOAD MODEL (CACHE)
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

    # Texture variation
    std = np.std(gray)

    # Channel similarity (MRI = grayscale)
    b, g, r = cv2.split(img)
    diff_rg = np.mean(np.abs(r - g))
    diff_rb = np.mean(np.abs(r - b))
    diff_gb = np.mean(np.abs(g - b))

    # Final condition
    if (diff_rg < 10 and diff_rb < 10 and diff_gb < 10) and (std < 80):
        return True
    else:
        return False

# -------------------------------
# FEATURE EXTRACTION FUNCTION
# -------------------------------
def extract_features(img):
    img_resized = cv2.resize(img, (64, 64))

    pixel_features = img_resized.flatten()

    mean = np.mean(img_resized)
    std = np.std(img_resized)
    maxv = np.max(img_resized)
    minv = np.min(img_resized)

    features = np.hstack([pixel_features, mean, std, maxv, minv])
    return features.reshape(1, -1)

# -------------------------------
# FILE UPLOAD
# -------------------------------
file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "png", "jpeg"])

if file is not None:

    # Load image
    img_pil = Image.open(file).convert("RGB")
    st.image(img_pil, caption="Uploaded Image", use_container_width=True)

    # Convert to numpy
    img = np.array(img_pil)

    # Extract features
    features = extract_features(img)

    # -------------------------------
    # PREDICT BUTTON
    # -------------------------------
    if st.button("Predict"):

        # STEP 1: MRI VALIDATION
        if not is_mri_image(img):
            st.error("❌ Invalid Image: Please upload a Brain MRI scan")
        
        else:
            # STEP 2: MODEL PREDICTION
            with st.spinner("Analyzing MRI..."):

                proba = model.predict_proba(features)
                confidence = np.max(proba)
                pred = np.argmax(proba)

            # STEP 3: LOW CONFIDENCE CHECK
            if confidence < 0.75:
                st.warning("⚠️ Low confidence prediction. Try another MRI image.")

            # STEP 4: RESULT
            if pred == 0:
                result = "Hemorrhagic Stroke"
            elif pred == 1:
                result = "Ischemic Stroke"
            else:
                result = "Unknown"

            # STEP 5: DISPLAY
            st.success(f"🧠 Prediction: {result}")
            st.write(f"Confidence: {confidence*100:.2f}%")
            st.progress(int(confidence * 100))
