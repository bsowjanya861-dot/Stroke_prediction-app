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
st.markdown("Upload a **Brain MRI image** to predict stroke type")

# -------------------- SIDEBAR --------------------
st.sidebar.title("About")
st.sidebar.info(
    "This app uses an XGBoost model to detect stroke type.\n\n"
    "Classes:\n"
    "- Hemorrhagic Stroke\n"
    "- Ischemic Stroke\n\n"
    "⚠️ Demo only (not medical use)"
)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("hybrid_stroke_model.json")
    return model

model = load_model()

# -------------------- FILE UPLOAD --------------------
file = st.file_uploader("📤 Upload Brain MRI Image", type=["jpg", "png", "jpeg"])

# -------------------- MAIN LOGIC --------------------
if file is not None:
    try:
        # Read image
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Convert to array
        img = np.array(img)

        # Safety check
        if img is None:
            st.error("❌ Invalid image file")
            st.stop()

        # Resize (same as training)
        img = cv2.resize(img, (64, 64))

        # Normalize
        img = img / 255.0

        # -------------------- IMAGE VALIDATION --------------------
        mean_val = np.mean(img)
        std_val = np.std(img)

        # Reject very dark/bright images
        if mean_val < 0.05 or mean_val > 0.95:
            st.error("❌ Invalid Image (Lighting issue / Not MRI)")
            st.stop()

        # Reject flat images
        if std_val < 0.05:
            st.error("❌ Invalid Image (Too plain / Not MRI)")
            st.stop()

        # -------------------- FEATURE EXTRACTION (IMPORTANT FIX) --------------------
        pixel_features = img.flatten()

        mean = np.mean(img)
        std = np.std(img)
        maxv = np.max(img)
        minv = np.min(img)

        features = np.hstack([pixel_features, mean, std, maxv, minv])
        features = features.reshape(1, -1)

        # -------------------- PREDICTION --------------------
        if st.button("🔍 Predict"):

            proba = model.predict_proba(features)
            confidence = float(np.max(proba))
            pred = int(np.argmax(proba))

            # Strong validation
            if confidence < 0.85:
                st.error("❌ Invalid Image (Not a Brain MRI)")
            else:
                if pred == 0:
                    st.error("⚠️ Hemorrhagic Stroke Detected")
                else:
                    st.success("✅ Ischemic Stroke Detected")

                st.write(f"Confidence: {confidence:.2f}")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
