import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier
import cv2
import joblib

st.set_page_config(page_title="Stroke Prediction", layout="centered")
st.title("🧠 Hybrid CNN + XGBoost Brain Stroke Prediction")

# Load CNN for feature extraction
cnn_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Load XGBoost stroke classifier
xgb_model = XGBClassifier()
xgb_model.load_model("hybrid_stroke_model.json")  # your existing XGBoost

# File uploader
file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "png", "jpeg"])

if file is not None:
    try:
        # Load image
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Resize for CNN
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Extract CNN features
        features = cnn_model.predict(img_array)

        # Predict with XGBoost
        if st.button("Predict"):
            pred = xgb_model.predict(features)
            if pred[0] == 0:
                st.error("⚠️ Hemorrhagic Stroke Detected")
            else:
                st.success("✅ Ischaemic Stroke Detected")

    except:
        st.warning("⚠️ Invalid image. Please upload a brain MRI image only.")
