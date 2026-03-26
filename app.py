import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from xgboost import XGBClassifier
import joblib  # to load saved XGBoost model

st.set_page_config(page_title="Stroke Prediction", layout="centered")
st.title("🧠 Hybrid CNN + XGBoost for Brain Stroke Prediction")

# Load pretrained CNN (MobileNetV2) as feature extractor
cnn_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# Load trained XGBoost classifier (trained on CNN features)
xgb_model = joblib.load("hybrid_stroke_xgb.pkl")  # save your trained XGB as .pkl

file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "png", "jpeg"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded MRI", use_container_width=True)

    # Preprocess image for CNN
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Extract features using CNN
    features = cnn_model.predict(img_array)

    if st.button("Predict"):
        # Predict using XGBoost on CNN features
        pred = xgb_model.predict(features)
        if pred[0] == 0:
            st.error("⚠️ Hemorrhagic Stroke Detected")
        elif pred[0]==1:
            st.success("✅ Ischaemic Stroke Detected")
        else:
            st.success("Unknown")
