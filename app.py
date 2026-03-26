import streamlit as st
import numpy as np
from PIL import Image
from xgboost import XGBClassifier

st.set_page_config(page_title="CT Stroke Prediction", layout="centered")
st.title("🧠 XGBoost Brain Stroke Prediction (RGB CT)")

# Load the trained XGBoost model
model = XGBClassifier()
model.load_model("hybrid_stroke_model.json")

def preprocess_image(img):
    # Resize to 224x224 (matches training)
    img = img.resize((224, 224))
    # Convert to RGB (if not already)
    img = img.convert("RGB")
    # Convert to NumPy array
    arr = np.array(img) / 255.0  # normalize
    # Flatten
    features = arr.flatten()
    return features.reshape(1, -1)

file = st.file_uploader("Upload CT scan", type=["jpg","png","jpeg"])

if file is not None:
    try:
        img = Image.open(file)
        st.image(img, caption="Uploaded CT Scan", use_container_width=True)

        # Preprocess
        features = preprocess_image(img)

        if st.button("Predict"):
            pred = model.predict(features)
            if pred[0] == 0:
                st.error("⚠️ Hemorrhagic Stroke Detected")
            else:
                st.success("✅ Ischaemic Stroke Detected")

    except:
        st.warning("⚠️ Invalid image. Please upload a CT scan only.")
