import streamlit as st
import numpy as np
import cv2
from PIL import Image
from xgboost import XGBClassifier

st.set_page_config(page_title="CT Stroke Prediction", layout="centered")
st.title("🧠 XGBoost Brain Stroke Prediction (CT Scan)")

# Load the trained XGBoost model
model = XGBClassifier()
model.load_model("hybrid_stroke_model.json")

# File uploader
file = st.file_uploader("Upload CT Scan Image", type=["jpg", "png", "jpeg"])

if file is not None:
    try:
        # Open and convert image
        img = Image.open(file).convert("L")  # grayscale CT
        st.image(img, caption="Uploaded CT Scan", use_container_width=True)

        # Resize to 64x64 (matches training)
        img_resized = np.array(img.resize((64, 64)))

        # Normalize pixel values
        img_normalized = img_resized / 255.0

        # Flatten + extra stats
        pixel_features = img_normalized.flatten()
        mean = np.mean(img_normalized)
        std = np.std(img_normalized)
        maxv = np.max(img_normalized)
        minv = np.min(img_normalized)
        features = np.hstack([pixel_features, mean, std, maxv, minv]).reshape(1, -1)

        # Predict
        if st.button("Predict"):
            pred = model.predict(features)
            if pred[0] == 0:
                st.error("⚠️ Hemorrhagic Stroke Detected")
            else:
                st.success("✅ Ischaemic Stroke Detected")

    except Exception as e:
        st.warning("⚠️ Invalid image. Please upload a CT scan only.")
