import streamlit as st
import numpy as np
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
        # Open image and convert to grayscale
        img = Image.open(file).convert("L")  # L = grayscale
        st.image(img, caption="Uploaded CT Scan", use_container_width=True)

        # Resize image to 64x64 (match training)
        img_resized = img.resize((64, 64))
        img_array = np.array(img_resized, dtype=np.float32)

        # Normalize pixel values to 0-1
        img_normalized = img_array / 255.0

        # Flatten and compute extra features
        pixel_features = img_normalized.flatten()
        mean = np.mean(img_normalized)
        std = np.std(img_normalized)
        maxv = np.max(img_normalized)
        minv = np.min(img_normalized)
        features = np.hstack([pixel_features, mean, std, maxv, minv]).reshape(1, -1)

        # Predict stroke type
        if st.button("Predict"):
            pred = model.predict(features)
            if pred[0] == 0:
                st.error("⚠️ Hemorrhagic Stroke Detected")
            else:
                st.success("✅ Ischaemic Stroke Detected")

    except Exception as e:
        st.warning("⚠️ Error processing image. Please upload a valid CT scan.")
        st.text(f"Debug info: {e}")
