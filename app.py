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
        # Open and convert image to RGB
        img = Image.open(file).convert("RGB")
        
        # Basic validation: image too small might not be MRI
        if img.size[0] < 50 or img.size[1] < 50:
            st.warning("⚠️ Please upload a valid brain MRI image.")
        else:
            st.image(img, caption="Uploaded Image", use_container_width=True)

            # Resize to 64x64 (must match model training)
            img = np.array(img)
            img = cv2.resize(img, (64, 64))

            # Normalize pixel values (0-255 → 0-1)
            img = img / 255.0

            # Extract features
            pixel_features = img.flatten()
            mean = np.mean(img)
            std = np.std(img)
            maxv = np.max(img)
            minv = np.min(img)
            features = np.hstack([pixel_features, mean, std, maxv, minv])
            features = features.reshape(1, -1)

            # Predict button
            if st.button("Predict"):
                pred = model.predict(features)
                if pred[0] == 0:
                    st.error("⚠️ Hemorrhagic Stroke Detected")
                else:
                    st.success("✅ Ischaemic Stroke Detected")
    except:
        st.warning("⚠️ Invalid image file. Please upload a JPG or PNG brain MRI image.")
