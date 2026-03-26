import streamlit as st
import numpy as np
import cv2
from PIL import Image
from xgboost import XGBClassifier

st.set_page_config(page_title="Stroke Prediction", layout="centered")
st.title("🧠 Hybrid XGBoost for Brain Stroke Prediction")

# Load model
model = XGBClassifier()
model.load_model("hybrid_stroke_model.json")  # Make sure this file is in the same folder

# Upload image
file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "png", "jpeg"])

if file is not None:
    # Open and display the image
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded MRI", use_container_width=True)

    # Preprocess for model
    img_array = np.array(img)
    img_array = cv2.resize(img_array, (64, 64))
    pixel_features = img_array.flatten()
    mean = np.mean(img_array)
    std = np.std(img_array)
    maxv = np.max(img_array)
    minv = np.min(img_array)
    features = np.hstack([pixel_features, mean, std, maxv, minv]).reshape(1, -1)

    if st.button("Predict"):
        # Make prediction
        pred = model.predict(features)
        proba = model.predict_proba(features)
        confidence = np.max(proba)

        if confidence < 0.6:
            st.error("⚠️ Invalid image. Please upload a proper brain MRI scan.")
        else:
            # Map prediction to label and image path
            stroke_map = {
                0: ("Hemorrhagic Stroke", "images/hemorrhagic.jpg"),
                1: ("Ischemic Stroke", "images/ischemic.jpg"),
                2: ("No Stroke", "images/normal.jpg")
            }

            result, image_path = stroke_map.get(pred[0], ("Unknown", None))

            # Display result
            st.success(f"{result} (Confidence: {confidence*100:.2f}%)")

            # Display only the corresponding image
            if image_path is not None:
                st.image(image_path, caption=result, use_column_width=True)
