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
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    mean = np.mean(gray)
    std = np.std(gray)

    # Heuristic rules (tune if needed)
    if std < 20:
        return False
    if mean < 10 or mean > 220:
        return False

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

        # -------------------- VALIDATION --------------------
        if not is_valid_mri(img):
            st.error("❌ Invalid Image: Not a Brain MRI")
        else:
            # Normalize
            img_norm = img / 255.0

            # Feature extraction
            pixel_features = img_norm.flatten()

            mean = np.mean(img_norm)
            std = np.std(img_norm)
            maxv = np.max(img_norm)
            minv = np.min(img_norm)

            features = np.hstack([pixel_features, mean, std, maxv, minv])
            features = features.reshape(1, -1)

            # -------------------- PREDICT --------------------
            if st.button("🔍 Predict"):

                proba = model.predict_proba(features)
                confidence = float(np.max(proba))
                pred = int(np.argmax(proba))

                # Confidence check
                if confidence < 0.90:
                    st.error("❌ Invalid Image: Low confidence. Please upload a valid Brain MRI scan.")
                    st.write(f"Confidence: {confidence:.2f}")
                else:
                    if pred == 0:
                        st.error("⚠️ Hemorrhagic Stroke Detected")
                    else:
                        st.success("✅ Ischemic Stroke Detected")

                    st.write(f"Confidence: {confidence:.2f}")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
