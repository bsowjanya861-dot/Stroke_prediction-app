import streamlit as st
import numpy as np
import cv2
from PIL import Image
from xgboost import XGBClassifier

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="CT Brain Stroke Detection",
    page_icon="🧠",
    layout="centered"
)

# -------------------- TITLE --------------------
st.title("🧠 CT Scan Stroke Detection")
st.markdown("Upload a **Brain CT scan image** to detect stroke")

# -------------------- SIDEBAR --------------------
st.sidebar.title("About")
st.sidebar.info(
    "This app detects stroke from CT scan images.\n\n"
    "⚠️ Only CT scans are supported.\n"
    "Other images will be rejected."
)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("ct_stroke_model.json")   # CT model only
    return model

model = load_model()

# -------------------- CT VALIDATION FUNCTION --------------------
def is_ct_scan(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    mean = np.mean(gray)
    std = np.std(gray)

    # CT scan characteristics:
    # - medium brightness
    # - moderate contrast
    if 50 < mean < 180 and std > 20:
        return True
    return False

# -------------------- FEATURE EXTRACTION --------------------
def extract_features(img):
    img = cv2.resize(img, (64, 64))

    pixel_features = img.flatten()

    mean = np.mean(img)
    std = np.std(img)
    maxv = np.max(img)
    minv = np.min(img)

    features = np.hstack([pixel_features, mean, std, maxv, minv])
    return features.reshape(1, -1)

# -------------------- FILE UPLOAD --------------------
file = st.file_uploader("📤 Upload CT Scan", type=["jpg", "png", "jpeg"])

# -------------------- MAIN --------------------
if file is not None:
    try:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        img = np.array(img)

        # -------------------- VALIDATE CT --------------------
        if not is_ct_scan(img):
            st.error("❌ Invalid Image: Please upload a brain CT scan")
        
        else:
            features = extract_features(img)

            if st.button("🔍 Predict"):

                proba = model.predict_proba(features)
                confidence = float(np.max(proba))
                pred = int(np.argmax(proba))

                # -------------------- CONFIDENCE CHECK --------------------
                if confidence < 0.80:
                    st.warning("⚠️ Low confidence. Image may not be clear CT scan.")
                    st.write(f"Confidence: {confidence:.2f}")

                else:
                    if pred == 0:
                        st.error("⚠️ Hemorrhagic Stroke Detected")
                    else:
                        st.success("✅ Ischemic Stroke Detected")

                    st.write(f"Confidence: {confidence:.2f}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
