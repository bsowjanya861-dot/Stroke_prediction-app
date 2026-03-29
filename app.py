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
    "⚠️ This is a demo project (not for medical use)"
)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("hybrid_stroke_model.json")
    return model

model = load_model()

# -------------------- IMAGE VALIDATION FUNCTION --------------------
def is_valid_mri(img):
    """
    Basic heuristic to reject non-MRI images
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    # Heuristic rules
    if std_intensity < 10:
        return False  # too flat / no structure

    if mean_intensity < 20 or mean_intensity > 230:
        return False  # too dark or too bright

    return True

# -------------------- FILE UPLOAD --------------------
file = st.file_uploader("📤 Upload Brain MRI Image", type=["jpg", "png", "jpeg"])

# -------------------- MAIN LOGIC --------------------
if file is not None:
    try:
        # Display image
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Convert to array
        img = np.array(img)

        # Resize
        img = cv2.resize(img, (64, 64))

        # -------------------- VALIDATION STEP --------------------
        if not is_valid_mri(img):
            st.error("❌ Invalid Image: Please upload a valid Brain MRI scan")
        else:
            # Feature extraction
            pixel_features = img.flatten()

            mean = np.mean(img)
            std = np.std(img)
            maxv = np.max(img)
            minv = np.min(img)

            features = np.hstack([pixel_features, mean, std, maxv, minv])
            features = features.reshape(1, -1)

            # -------------------- PREDICT --------------------
            if st.button("🔍 Predict"):
                proba = model.predict_proba(features)
                confidence = float(np.max(proba))
                pred = int(np.argmax(proba))

                # Confidence filter
                if confidence < 0.90:
                    st.error("❌ Invalid Image: Confidence too low. Please upload a valid Brain MRI scan.")
                    st.write(f"Confidence: {confidence:.2f}")
                else:
                    if pred == 0:
                        st.error("⚠️ Hemorrhagic Stroke Detected")
                    else:
                        st.success("✅ Ischemic Stroke Detected")

                    st.write(f"Confidence: {confidence:.2f}")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
