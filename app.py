import streamlit as st
import numpy as np
import cv2
from PIL import Image
from xgboost import XGBClassifier

st.set_page_config(page_title="Stroke Prediction", layout="centered")

st.title("🧠 Hybrid XGBoost for Brain Stroke Prediction")

model = XGBClassifier()
model.load_model("hybrid_stroke_model.json")

file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "png", "jpeg"])

if file is not None:

    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded MRI", use_container_width=True)

    img = np.array(img)
    img = cv2.resize(img, (64, 64))

    pixel_features = img.flatten()

    mean = np.mean(img)
    std = np.std(img)
    maxv = np.max(img)
    minv = np.min(img)

    features = np.hstack([pixel_features, mean, std, maxv, minv])
    features = features.reshape(1, -1)

    if st.button("Predict"):
        # Predict
        pred = model.predict(features)
        proba = model.predict_proba(features)
        confidence = np.max(proba)

        # Check for invalid / non-MRI images
        if confidence < 0.6:
            st.error("⚠️ Invalid image. Please upload a proper brain MRI scan.")
        else:
            if pred[0] == 0:
                result = "Hemorrhagic Stroke"
            elif pred[0] == 1:
                result = "Ischemic Stroke"
            else:
                result = "No Stroke"

            st.success(f"{result} (Confidence: {confidence*100:.2f}%)")
