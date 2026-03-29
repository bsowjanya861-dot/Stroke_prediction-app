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
    "⚠️ Demo project (not medical use)"
)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("hybrid_stroke_model.json")
    return model

model = load_model()

# -------------------- STRONG VALIDATION --------------------
def is_valid_mri(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    # Edge detection (structure check)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges)

    # Circular structure approximation (brain shape tendency)
    h, w = gray.shape
    center_region = gray[h//4:3*h//4, w//4:3*w//4]
    center_mean = np.mean(center_region)

    # Heuristics
    if std_intensity < 15:
        return False

    if edge_density < 5:
        return False

    if mean_intensity < 30 or mean_intensity > 220:
        return False

    if center_mean < 20:
        return False

    return True

# -------------------- FILE UPLOAD --------------------
file = st.file_uploader("📤 Upload Brain MRI Image", type=["jpg", "png", "jpeg"])

# -------------------- MAIN LOGIC --------------------
if file is not None:
    try:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        img = np.array(img)
        img = cv2.resize(img, (64, 64))

        # -------------------- VALIDATION --------------------
        if not is_valid_mri(img):
            st.error("❌ Invalid Image: Not a Brain MRI")
        else:
            pixel_features = img.flatten()

            mean = np.mean(img)
            std = np.std(img)
            maxv = np.max(img)
            minv = np.min(img)

            features = np.hstack([pixel_features, mean, std, maxv, minv])
            features = features.reshape(1, -1)

            if st.button("🔍 Predict"):

                proba = model.predict_proba(features)
                sorted_probs = np.sort(proba[0])[::-1]

                confidence = float(sorted_probs[0])
                gap = sorted_probs[0] - sorted_probs[1]

                # -------------------- STRICT FILTER --------------------
                if confidence < 0.95 or gap < 0.2:
                    st.error("❌ Uncertain or invalid prediction. Upload a clearer Brain MRI.")
                    st.write(f"Confidence: {confidence:.2f}")
                    st.write(f"Probability gap: {gap:.2f}")
                else:
                    pred = int(np.argmax(proba))

                    if pred == 0:
                        st.error("⚠️ Hemorrhagic Stroke Detected")
                    else:
                        st.success("✅ Ischemic Stroke Detected")

                    st.write(f"Confidence: {confidence:.2f}")
                    st.write(f"Probability gap: {gap:.2f}")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
