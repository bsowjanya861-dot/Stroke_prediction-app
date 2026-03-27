import streamlit as st
import numpy as np
import cv2
from PIL import Image
from xgboost import XGBClassifier

# ---------------- PAGE ----------------
st.set_page_config(page_title="Brain Stroke Detection", layout="centered")

st.title("🧠 Brain Stroke Detection App")
st.markdown("Upload a brain scan image")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("hybrid_stroke_model.json")
    return model

model = load_model()

# ---------------- UPLOAD ----------------
file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file is not None:
    try:
        img = Image.open(file).convert("RGB")
        st.image(img, use_container_width=True)

        img = np.array(img)
        img = cv2.resize(img, (64, 64))

        # Feature extraction
        pixel_features = img.flatten()

        mean = np.mean(img)
        std = np.std(img)
        maxv = np.max(img)
        minv = np.min(img)

        features = np.hstack([pixel_features, mean, std, maxv, minv])
        features = features.reshape(1, -1)

        if st.button("Predict"):

            proba = model.predict_proba(features)
            confidence = float(np.max(proba))
            pred = int(np.argmax(proba))

            # ---------------- SMART VALIDATION ----------------
            if confidence < 0.65:
                st.error("❌ Invalid Image (Not a brain scan)")
                st.write(f"Confidence: {confidence:.2f}")

            elif confidence < 0.75:
                st.warning("⚠️ Low confidence prediction (image unclear)")
                st.write(f"Confidence: {confidence:.2f}")

                if pred == 0:
                    st.error("⚠️ Hemorrhagic Stroke (low confidence)")
                else:
                    st.success("✅ Ischemic Stroke (low confidence)")

            else:
                # HIGH CONFIDENCE
                if pred == 0:
                    st.error("⚠️ Hemorrhagic Stroke Detected")
                else:
                    st.success("✅ Ischemic Stroke Detected")

                st.write(f"Confidence: {confidence:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
