import streamlit as st
import numpy as np
import cv2
from PIL import Image
from xgboost import XGBClassifier

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Stroke Prediction", layout="centered")

st.title("🧠 Hybrid XGBoost for Brain Stroke Prediction")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("hybrid_stroke_model.json")
    return model

model = load_model()

# -------------------- IMAGE VALIDATION --------------------
def is_valid_mri(img):
    """
    Heuristic checks to filter non-brain / invalid images
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    # Too flat → likely not MRI
    if std_intensity < 10:
        return False

    # Too dark or too bright
    if mean_intensity < 20 or mean_intensity > 230:
        return False

    # Check edge density (MRI has structural edges)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])

    if edge_density < 0.01:
        return False

    return True

# -------------------- UPLOAD --------------------
file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "png", "jpeg"])

if file is not None:

    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = np.array(img)
    img = cv2.resize(img, (64, 64))

    # -------------------- VALIDATION STEP --------------------
    if not is_valid_mri(img):
        st.error("❌ Invalid Image: Not a valid Brain MRI scan")
    else:
        # -------------------- FEATURE EXTRACTION --------------------
        img_norm = img / 255.0

        pixel_features = img_norm.flatten()

        mean = np.mean(img_norm)
        std = np.std(img_norm)
        maxv = np.max(img_norm)
        minv = np.min(img_norm)

        features = np.hstack([pixel_features, mean, std, maxv, minv])
        features = features.reshape(1, -1)

        # -------------------- PREDICTION --------------------
        if st.button("Predict"):

            proba = model.predict_proba(features)
            confidence = float(np.max(proba))
            pred = int(np.argmax(proba))

            # Confidence threshold
            if confidence < 0.80:
                st.error("❌ Low confidence. Image may be invalid or unclear.")
                st.write(f"Confidence: {confidence:.2f}")
            else:
                if pred == 0:
                    st.error("⚠️ Hemorrhagic Stroke Detected")
                else:
                    st.success("✅ Ischemic Stroke Detected")

                st.write(f"Confidence: {confidence:.2f}")
