import streamlit as st
import numpy as np
import cv2
from PIL import Image
from xgboost import XGBClassifier

st.set_page_config(page_title="Stroke Detection", layout="centered")

st.title("🧠 Stroke Detection")

@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("hybrid_stroke_model.json")
    return model

model = load_model()

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file is not None:

    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)

    img = np.array(img)
    img = cv2.resize(img, (64, 64))

    pixel_features = img.flatten()

    mean = np.mean(img)
    std = np.std(img)
    maxv = np.max(img)
    minv = np.min(img)

    features = np.hstack([pixel_features, mean, std, maxv, minv])
    features = features.reshape(1, -1)

    if st.button("🔍 Predict"):

        proba = model.predict_proba(features)
        confidence = float(np.max(proba))
        pred = int(np.argmax(proba))

        if confidence < 0.60:
            st.error("❌ Invalid Image")
        elif confidence < 0.80:
            st.warning("⚠️ Low Confidence")

        if pred == 0:
            st.error("⚠️ Hemorrhagic Stroke")
        else:
            st.success("✅ Ischemic Stroke")

        st.write(f"Confidence: {confidence:.2f}")
