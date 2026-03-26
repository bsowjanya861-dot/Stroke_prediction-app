import streamlit as st
import numpy as np
import cv2
from PIL import Image
from xgboost import XGBClassifier

st.set_page_config(page_title="Stroke Prediction", layout="centered")
st.title("🧠 Hybrid XGBoost for Brain Stroke Prediction")

# Load your pretrained XGBoost model
model = XGBClassifier()
model.load_model("hybrid_stroke_model.json")

# File uploader
file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "png", "jpeg"])

if file is not None:
    try:
        # Open image
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Resize to 64x64 (matches model training)
        img = img.resize((64, 64))
        img_array = np.array(img)

        # Normalize pixel values
        img_array = img_array / 255.0

        # Flatten + extra stats
        pixel_features = img_array.flatten()
        mean = np.mean(img_array)
        std = np.std(img_array)
        maxv = np.max(img_array)
        minv = np.min(img_array)
        features = np.hstack([pixel_features, mean, std, maxv, minv]).reshape(1, -1)

        # Predict button
        if st.button("Predict"):
            pred = model.predict(features)
            if pred[0] == 0:
                st.error("⚠️ Hemorrhagic Stroke Detected")
            else:
                st.success("✅ Ischaemic Stroke Detected")

    except Exception as e:
        st.warning("⚠️ Invalid image. Please upload a brain MRI image only.")
