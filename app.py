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
st.markdown("Upload a **Brain Scan Image (CT/MRI)** to predict stroke type")

# -------------------- SIDEBAR --------------------
st.sidebar.title("About")
st.sidebar.info(
    "This app uses an XGBoost model to detect stroke type.\n\n"
    "Classes:\n"
    "- Hemorrhagic Stroke\n"
    "- Ischemic Stroke\n"
    "- Normal\n\n"
    "⚠️ Demo project (not for medical use)"
)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("hybrid_stroke_model.json")  # must be trained with 3 classes
    return model

model = load_model()

# -------------------- FILE UPLOAD --------------------
file = st.file_uploader("📤 Upload Brain Image", type=["jpg", "png", "jpeg"])

# -------------------- FEATURE FUNCTION --------------------
def extract_features(img):
    img = cv2.resize(img, (64, 64))
    img = img / 255.0   # normalize

    pixel_features = img.flatten()

    mean = np.mean(img)
    std = np.std(img)
    maxv = np.max(img)
    minv = np.min(img)

    features = np.hstack([pixel_features, mean, std, maxv, minv])
    return features.reshape(1, -1)

# -------------------- MAIN LOGIC --------------------
if file is not None:
    try:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        img = np.array(img)

        if st.button("🔍 Predict"):

            features = extract_features(img)

            proba = model.predict_proba(features)
            confidence = float(np.max(proba))
            pred = int(np.argmax(proba))

            # -------------------- LOW CONFIDENCE CHECK --------------------
            if confidence < 0.75:
                st.warning("⚠️ Low confidence prediction")
                st.info("👉 Image may be unclear or not a valid brain scan")
                st.write(f"Confidence: {confidence:.2f}")

            else:
                # -------------------- OUTPUT --------------------
                if pred == 0:
                    st.error("⚠️ Hemorrhagic Stroke Detected")
                elif pred == 1:
                    st.success("✅ Ischemic Stroke Detected")
                elif pred == 2:
                    st.info("🟢 Normal (No Stroke Detected)")

                st.write(f"Confidence: {confidence:.2f}")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
