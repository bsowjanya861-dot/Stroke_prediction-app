import streamlit as st
import numpy as np
import cv2
from PIL import Image
from xgboost import XGBClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Brain Stroke Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Stroke Detection")
st.markdown("Upload a **Brain Scan Image (CT/MRI only)**")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("hybrid_stroke_model.json")
    return model

model = load_model()

# ---------------- FILE ----------------
file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])


# ---------------- FEATURE ----------------
def extract_features(img):
    img = cv2.resize(img, (64, 64))
    img = img / 255.0

    pixel = img.flatten()
    mean = np.mean(img)
    std = np.std(img)
    maxv = np.max(img)
    minv = np.min(img)

    return np.hstack([pixel, mean, std, maxv, minv]).reshape(1, -1)


# ---------------- BRAIN CHECK (IMPORTANT) ----------------
def is_brain_scan(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    brightness = np.mean(gray)

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])

    # Brain CT/MRI pattern:
    # medium brightness + visible skull edges
    if 60 < brightness < 180 and edge_density > 3:
        return True
    return False


# ---------------- MAIN ----------------
if file is not None:
    try:
        img = Image.open(file).convert("RGB")
        st.image(img, use_container_width=True)

        img = np.array(img)

        if st.button("🔍 Predict"):

            # STEP 1: VALIDATE IMAGE
            if not is_brain_scan(img):
                st.error("❌ Invalid Image: Please upload a Brain CT/MRI scan only")
            
            else:
                # STEP 2: PREDICT
                features = extract_features(img)

                proba = model.predict_proba(features)
                confidence = float(np.max(proba))
                pred = int(np.argmax(proba))

                if confidence < 0.65:
                    st.warning("⚠️ Low confidence prediction")

                if pred == 0:
                    st.error("⚠️ Hemorrhagic Stroke Detected")
                else:
                    st.success("✅ Ischemic Stroke Detected")

                st.write(f"Confidence: {confidence:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
