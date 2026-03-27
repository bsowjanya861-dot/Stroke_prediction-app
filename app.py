import streamlit as st
import numpy as np
import cv2
from PIL import Image
from xgboost import XGBClassifier

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Brain Stroke Detection",
    page_icon="🧠",
    layout="centered"
)

# -------------------- TITLE --------------------
st.title("🧠 Brain Stroke Detection App")
st.markdown("Upload a **Brain CT/MRI image** to predict stroke type")

# -------------------- SIDEBAR --------------------
st.sidebar.title("About")
st.sidebar.info(
    "Detects stroke from brain scans.\n\n"
    "⚠️ Only brain scans are supported\n"
    "⚠️ Demo project"
)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("hybrid_stroke_model.json")
    return model

model = load_model()

# -------------------- FILE --------------------
file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

# -------------------- FEATURE FUNCTION --------------------
def extract_features(img):
    img = cv2.resize(img, (64, 64))
    img = img / 255.0

    pixel = img.flatten()
    mean = np.mean(img)
    std = np.std(img)
    maxv = np.max(img)
    minv = np.min(img)

    return np.hstack([pixel, mean, std, maxv, minv]).reshape(1, -1)

# -------------------- IMAGE VALIDATION (KEY FIX) --------------------
def is_brain_scan(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    brightness = np.mean(gray)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])

    # Brain scans:
    # - medium brightness
    # - strong circular edges (skull)
    if 60 < brightness < 180 and edge_density > 5:
        return True
    return False

# -------------------- MAIN --------------------
if file is not None:
    try:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        img = np.array(img)

        if st.button("🔍 Predict"):

            # STEP 1: VALIDATE IMAGE
            if not is_brain_scan(img):
                st.error("❌ Invalid Image: Not a brain scan")
            
            else:
                # STEP 2: PREDICT
                features = extract_features(img)

                proba = model.predict_proba(features)
                confidence = float(np.max(proba))
                pred = int(np.argmax(proba))

                # STEP 3: CONFIDENCE CHECK
                if confidence < 0.70:
                    st.warning("⚠️ Low confidence prediction")
                
                # RESULT
                if pred == 0:
                    st.error("⚠️ Hemorrhagic Stroke Detected")
                else:
                    st.success("✅ Ischemic Stroke Detected")

                st.write(f"Confidence: {confidence:.2f}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
