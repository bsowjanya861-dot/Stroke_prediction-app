import streamlit as st
import numpy as np
import cv2
from PIL import Image
from xgboost import XGBClassifier

# -------------------------------
# PAGE SETTINGS
# -------------------------------
st.set_page_config(page_title="Stroke Prediction", layout="centered")
st.title("🧠 Brain Stroke Prediction (MRI or CT)")

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
# VALID BRAIN SCAN CHECK FUNCTION
# -------------------------------
def is_valid_brain_scan(img):
    """Check if the uploaded image looks like a brain scan (MRI or CT)"""
    img = np.array(img)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Check contrast/edges (CT has more high contrast)
    std = np.std(gray)
    mean = np.mean(gray)

    # Reject completely blank or colorful images
    if std < 5 or np.mean(img) > 250 or np.mean(img) < 5:
        return False
    return True

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
file = st.file_uploader("Upload Brain Scan (MRI or CT)", type=["jpg", "png", "jpeg"])

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

        # STEP 1: Validate brain scan
        if not is_valid_brain_scan(img):
            st.error("❌ Invalid Image: Please upload a Brain MRI or CT scan")
        
        else:
            # STEP 2: MODEL PREDICTION
            with st.spinner("Analyzing Brain Scan..."):
                proba = model.predict_proba(features)
                confidence = np.max(proba)
                pred = np.argmax(proba)

            # STEP 3: LOW CONFIDENCE CHECK
            if confidence < 0.7:
                st.warning("⚠️ Low confidence prediction. Try another scan image.")

            # STEP 4: RESULT
            if pred == 0:
                result = "Hemorrhagic Stroke"
            elif pred == 1:
                result = "Ischemic Stroke"
            else:
                result = "No Stroke Detected"

            # STEP 5: DISPLAY
            st.success(f"🧠 Prediction: {result}")
            st.write(f"Confidence: {confidence*100:.2f}%")
