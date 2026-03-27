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
st.markdown("Upload a **Brain Scan (MRI / CT)** to predict stroke type")

# -------------------- SIDEBAR --------------------
st.sidebar.title("About")
st.sidebar.info(
    "This app uses an XGBoost model to detect stroke type.\n\n"
    "Classes:\n"
    "- Hemorrhagic Stroke\n"
    "- Ischemic Stroke\n\n"
    "⚠️ Demo project (not for medical use)\n"
    "⚠️ Performance depends on training data"
)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("hybrid_stroke_model.json")
    return model

model = load_model()

# -------------------- FILE UPLOAD --------------------
file = st.file_uploader("📤 Upload Brain Scan Image", type=["jpg", "png", "jpeg"])

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

# -------------------- MAIN --------------------
if file is not None:
    try:
        # Show image
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        img = np.array(img)

        # Extract features
        features = extract_features(img)

        if st.button("🔍 Predict"):

            # Prediction
            proba = model.predict_proba(features)
            confidence = float(np.max(proba))
            pred = int(np.argmax(proba))

            # -------------------- SMART VALIDATION --------------------
            if confidence < 0.60:
                st.error("❌ Invalid Image")
                st.info("👉 This image does not match trained brain scan patterns")
            
            elif 0.60 <= confidence < 0.75:
                st.warning("⚠️ Low Confidence Prediction")
                st.info("👉 Model is unsure. Try a clearer CT/MRI image")

            else:
                # High confidence prediction
                if pred == 0:
                    st.error("⚠️ Hemorrhagic Stroke Detected")
                else:
                    st.success("✅ Ischemic Stroke Detected")

            # Always show confidence
            st.write(f"Confidence: {confidence:.2f}")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
