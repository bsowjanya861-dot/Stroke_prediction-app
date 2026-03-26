import streamlit as st
import numpy as np
import cv2
from PIL import Image
from xgboost import XGBClassifier

# -------------------------------
# PAGE SETTINGS
# -------------------------------
st.set_page_config(page_title="CT Stroke Prediction", layout="centered")
st.title("🧠 Brain CT Scan Stroke Prediction")

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
# CT SCAN CHECK FUNCTION
# -------------------------------
def is_ct_scan(img):
    """
    Detect if the uploaded image is likely a CT scan.
    CT scans are mostly grayscale, low color variation, and medium contrast.
    """
    img = np.array(img)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    std = np.std(gray)  # contrast

    # Channel similarity check (CT scans are grayscale)
    b, g, r = cv2.split(img)
    diff_rg = np.mean(np.abs(r - g))
    diff_rb = np.mean(np.abs(r - b))
    diff_gb = np.mean(np.abs(g - b))

    # Final rule
    if (diff_rg < 10 and diff_rb < 10 and diff_gb < 10) and (std < 80):
        return True
    else:
        return False

# -------------------------------
# FILE UPLOAD
# -------------------------------
file = st.file_uploader("Upload Brain CT Scan Image", type=["jpg", "png", "jpeg"])

if file is not None:
    # Show uploaded image
    img_pil = Image.open(file).convert("RGB")
    st.image(img_pil, caption="Uploaded Image", use_container_width=True)

    # Convert to numpy for processing
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

    features = np.hstack([pixel_features, mean, std, maxv, minv]).reshape(1, -1)

    # -------------------------------
    # PREDICTION BUTTON
    # -------------------------------
    if st.button("Predict"):

        # Step 1: Validate CT scan
        if not is_ct_scan(img):
            st.error("❌ This is NOT a valid Brain CT scan image. Please upload a proper CT scan.")
        else:
            # Step 2: Predict stroke type
            with st.spinner("Analyzing CT scan..."):
                proba = model.predict_proba(features)
                confidence = np.max(proba)
                pred = np.argmax(proba)

            # Step 3: Show result
            if pred == 0:
                result = "Hemorrhagic Stroke"
            elif pred == 1:
                result = "Ischemic Stroke"
            else:
                result = "Unknown"

            st.success(f"🧠 Prediction: {result}")
            st.write(f"Confidence: {confidence*100:.2f}%")
            st.progress(int(confidence * 100))
